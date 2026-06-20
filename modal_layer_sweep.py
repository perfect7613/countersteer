"""Train-only coarse-to-fine causal layer sweep on Modal."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import modal

from countersteer.artifacts import load_or_create_json_pair
from countersteer.corpus import canonical_hash
from countersteer.layer_sweep import (
    rank_layers,
    refinement_layers,
    regularly_spaced_layers,
    select_causal_layer,
    summarize_layer,
)
from countersteer.patching import (
    last_non_padding_index,
    restoration_fraction,
    select_patch_examples,
)
from countersteer.pressurebench import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    RENDER_VERSION,
    build_pressure_bundle,
)
from countersteer.provenance import CostRates, estimate_resource_cost_usd


APP_NAME = "countersteer-causal-layer-sweep"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
GPU = "L4"
CPU_CORES = 2.0
MEMORY_MIB = 32_768
VOLUME_PATH = Path("/countersteer")
LAYER_COUNT = 42
COARSE_LAYERS = regularly_spaced_layers(LAYER_COUNT, 8)
COARSE_CANDIDATES = 2
REFINEMENT_RADIUS = 2
SEED = 42
BOOTSTRAP_REPLICATES = 2000
MIN_PAIRED_ADVANTAGE = 0.05
TOKEN_POSITION = "last_non_padding_prompt_token_at_decoder_layer_output"
MODAL_RATES = CostRates(0.000222, 0.0000131, 0.00000222)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("countersteer-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("countersteer-huggingface")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "huggingface-hub==1.20.1",
        "safetensors==0.8.0",
        "torch==2.12.1",
        "transformers==5.12.1",
    )
    .add_local_python_source("countersteer")
)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=30 * 60,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    secrets=[hf_secret],
    volumes={str(VOLUME_PATH): volume},
)
@modal.concurrent(max_inputs=1)
class CausalLayerSweep:
    @modal.enter()
    def load(self) -> None:
        import os

        os.environ["HF_HOME"] = str(VOLUME_PATH / "hf-cache")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        core = getattr(self.model, "model", None)
        self.layers = getattr(core, "layers", None)
        if self.layers is None:
            self.layers = getattr(getattr(core, "language_model", None), "layers", None)
        if self.layers is None or len(self.layers) != LAYER_COUNT:
            raise RuntimeError(f"expected {LAYER_COUNT} Gemma 4 decoder layers")
        encoded = {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in ("A", "B")
        }
        if any(len(ids) != 1 for ids in encoded.values()):
            raise RuntimeError(f"A/B labels must tokenize once, got {encoded}")
        self.label_ids = {label: ids[0] for label, ids in encoded.items()}
        volume.commit()

    def _forward(
        self,
        row: dict,
        *,
        capture_all: bool = False,
        patch_layer: int | None = None,
        replacement=None,
    ) -> tuple[dict, dict[int, object]]:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        token_index = last_non_padding_index(inputs["attention_mask"][0].tolist())
        captured: dict[int, object] = {}
        handles = []

        if capture_all:
            for layer_index, layer in enumerate(self.layers):
                def capture_hook(_module, _inputs, output, index=layer_index):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[index] = hidden[:, token_index, :].detach().clone()

                handles.append(layer.register_forward_hook(capture_hook))

        if patch_layer is not None:
            if replacement is None:
                raise ValueError("patch replacement is required")

            def patch_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                patched[:, token_index, :] = replacement.to(hidden.device, hidden.dtype)
                return (patched, *output[1:]) if isinstance(output, tuple) else patched

            handles.append(self.layers[patch_layer].register_forward_hook(patch_hook))

        try:
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
        finally:
            for handle in handles:
                handle.remove()

        pair = torch.stack(
            [logits[:, self.label_ids["A"]], logits[:, self.label_ids["B"]]], dim=-1
        ).float()[0]
        correct_index = 0 if row["correct_label"] == "A" else 1
        wrong_index = 1 - correct_index
        score = {
            "predicted_label": "A" if pair[0] >= pair[1] else "B",
            "correct_logit_margin": float(pair[correct_index] - pair[wrong_index]),
            "token_index": token_index,
        }
        return score, captured

    @staticmethod
    def _random_replacement(wrong, neutral, seed: int):
        import torch

        wrong_float = wrong.float()
        delta = neutral.float() - wrong_float
        generator = torch.Generator(device=wrong.device)
        generator.manual_seed(seed)
        direction = torch.randn(
            wrong.shape, generator=generator, device=wrong.device, dtype=torch.float32
        )
        replacement = wrong_float + direction / direction.norm() * delta.norm()
        return replacement, float(delta.norm()), float((replacement - wrong_float).norm())

    def _load_bundle(self):
        import os

        from datasets import load_dataset

        dataset = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split="train",
            token=os.environ["HF_TOKEN"],
        )
        if len(dataset) != 194:
            raise RuntimeError(f"expected 194 PressureBench rows, found {len(dataset)}")
        pressure_rows = [dict(row) for row in dataset if row.get("question_syco")]
        if len(pressure_rows) != 40:
            raise RuntimeError(f"expected 40 pressure rows, found {len(pressure_rows)}")
        return build_pressure_bundle(pressure_rows)

    def _load_baseline(self, bundle) -> tuple[list[dict], str]:
        config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_hash": bundle.dataset_hash,
            "split_hash": bundle.split_hash,
            "conditions": ["neutral", "wrong_belief"],
            "scoring": "constrained-next-token-A-vs-B-v1",
        }
        config_hash = canonical_hash(config)
        item_dir = (
            VOLUME_PATH / "results" / "pressurebench-baseline" / config_hash / "items"
        )
        records = []
        for row in bundle.prompts:
            if row.condition == "correct_belief":
                continue
            path = item_dir / f"{row.item_id}.{row.condition}.json"
            if not path.exists():
                raise RuntimeError(
                    "pinned PressureBench baseline is missing; run modal_patching.py first"
                )
            records.append(json.loads(path.read_text("utf-8")))
        return records, config_hash

    def _capture_examples(self, selections, prompt_map):
        captures = {}
        for selection in selections:
            item_id = selection.item_id
            neutral_score, neutral = self._forward(
                prompt_map[(item_id, "neutral")], capture_all=True
            )
            wrong_score, wrong = self._forward(
                prompt_map[(item_id, "wrong_belief")], capture_all=True
            )
            if len(neutral) != LAYER_COUNT or len(wrong) != LAYER_COUNT:
                raise RuntimeError("not every decoder layer was captured")
            captures[item_id] = {
                "neutral_score": neutral_score,
                "wrong_score": wrong_score,
                "neutral": neutral,
                "wrong": wrong,
            }
        return captures

    def _run_layer(
        self,
        *,
        layer_index: int,
        phase: str,
        selections,
        prompt_map,
        captures,
        run_dir: Path,
    ) -> dict:
        artifact_path = run_dir / phase / f"layer-{layer_index:02d}.json"
        if artifact_path.exists():
            return json.loads(artifact_path.read_text("utf-8"))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for example_index, selection in enumerate(selections):
            item_id = selection.item_id
            unrelated_id = selections[(example_index + 1) % len(selections)].item_id
            wrong_row = prompt_map[(item_id, "wrong_belief")]
            source = captures[item_id]
            neutral_activation = source["neutral"][layer_index]
            wrong_activation = source["wrong"][layer_index]
            unrelated_activation = captures[unrelated_id]["neutral"][layer_index]
            random_seed = SEED + layer_index * 1000 + example_index
            random_activation, target_norm, random_norm = self._random_replacement(
                wrong_activation, neutral_activation, random_seed
            )
            matched_score, _ = self._forward(
                wrong_row, patch_layer=layer_index, replacement=neutral_activation
            )
            unrelated_score, _ = self._forward(
                wrong_row, patch_layer=layer_index, replacement=unrelated_activation
            )
            random_score, _ = self._forward(
                wrong_row, patch_layer=layer_index, replacement=random_activation
            )
            neutral_margin = source["neutral_score"]["correct_logit_margin"]
            wrong_margin = source["wrong_score"]["correct_logit_margin"]

            def fraction(score):
                value = restoration_fraction(
                    neutral_margin=neutral_margin,
                    wrong_margin=wrong_margin,
                    patched_margin=score["correct_logit_margin"],
                )
                if value is None:
                    raise RuntimeError("eligible example has invalid restoration denominator")
                return value

            rows.append(
                {
                    "item_id": item_id,
                    "layer_index": layer_index,
                    "neutral_margin": neutral_margin,
                    "wrong_margin": wrong_margin,
                    "matched_restoration_fraction": fraction(matched_score),
                    "unrelated_restoration_fraction": fraction(unrelated_score),
                    "random_restoration_fraction": fraction(random_score),
                    "matched_answer_restored": (
                        matched_score["predicted_label"] == wrong_row["correct_label"]
                    ),
                    "unrelated_answer_restored": (
                        unrelated_score["predicted_label"] == wrong_row["correct_label"]
                    ),
                    "random_answer_restored": (
                        random_score["predicted_label"] == wrong_row["correct_label"]
                    ),
                    "unrelated_item_id": unrelated_id,
                    "random_seed": random_seed,
                    "matched_delta_norm": target_norm,
                    "random_delta_norm": random_norm,
                    "random_norm_absolute_error": abs(target_norm - random_norm),
                }
            )
        summary = summarize_layer(
            layer_index,
            rows,
            seed=SEED,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
        )
        artifact = {
            "schema_version": 1,
            "phase": phase,
            "layer_index": layer_index,
            "token_position_semantics": TOKEN_POSITION,
            "summary": summary.as_dict(),
            "examples": rows,
        }
        with artifact_path.open("x", encoding="utf-8") as file:
            json.dump(artifact, file, indent=2)
            file.write("\n")
        volume.commit()
        return artifact

    @modal.method()
    def run(self) -> dict:
        import platform

        import torch

        experiment_started = time.perf_counter()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        bundle = self._load_bundle()
        baseline_records, baseline_hash = self._load_baseline(bundle)
        selections, eligible_count = select_patch_examples(
            baseline_records, bundle.train_ids, limit=len(bundle.train_ids)
        )
        if eligible_count < 2 or len(selections) != eligible_count:
            raise RuntimeError("layer sweep requires at least two genuine training flips")
        prompt_map = {(row.item_id, row.condition): asdict(row) for row in bundle.prompts}
        run_config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_hash": bundle.dataset_hash,
            "split_hash": bundle.split_hash,
            "source_baseline_hash": baseline_hash,
            "training_item_ids": [row.item_id for row in selections],
            "coarse_layers": list(COARSE_LAYERS),
            "coarse_candidates": COARSE_CANDIDATES,
            "refinement_radius": REFINEMENT_RADIUS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "minimum_paired_advantage": MIN_PAIRED_ADVANTAGE,
            "token_position_semantics": TOKEN_POSITION,
            "seed": SEED,
        }
        run_hash = canonical_hash(run_config)
        run_dir = VOLUME_PATH / "results" / "causal-layer-sweep" / run_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        captures = self._capture_examples(selections, prompt_map)

        coarse_artifacts = [
            self._run_layer(
                layer_index=layer,
                phase="coarse",
                selections=selections,
                prompt_map=prompt_map,
                captures=captures,
                run_dir=run_dir,
            )
            for layer in COARSE_LAYERS
        ]
        coarse_summaries = [
            summarize_layer(
                artifact["layer_index"],
                artifact["examples"],
                seed=SEED,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
            )
            for artifact in coarse_artifacts
        ]
        strongest_coarse = [
            row.layer_index for row in rank_layers(coarse_summaries)[:COARSE_CANDIDATES]
        ]
        fine_layers = refinement_layers(
            strongest_coarse,
            layer_count=LAYER_COUNT,
            coarse_layers=COARSE_LAYERS,
            radius=REFINEMENT_RADIUS,
        )
        fine_artifacts = [
            self._run_layer(
                layer_index=layer,
                phase="refinement",
                selections=selections,
                prompt_map=prompt_map,
                captures=captures,
                run_dir=run_dir,
            )
            for layer in fine_layers
        ]
        all_artifacts = coarse_artifacts + fine_artifacts
        summaries = [
            summarize_layer(
                artifact["layer_index"],
                artifact["examples"],
                seed=SEED,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
            )
            for artifact in all_artifacts
        ]
        ranked = rank_layers(summaries)
        selected = select_causal_layer(
            ranked, min_paired_advantage=MIN_PAIRED_ADVANTAGE
        )
        selected_layer = selected.layer_index if selected else None
        if selected:
            rationale = (
                f"Layer {selected_layer} has the strongest conservative paired advantage "
                "and matched patching exceeds both controls in mean restoration and "
                "discrete answer restoration under the predeclared threshold."
            )
        else:
            rationale = (
                "No swept layer cleared the predeclared matched-over-controls criteria; "
                "no intervention layer was selected."
            )

        compute_seconds = self.load_seconds + (time.perf_counter() - experiment_started)
        cost = estimate_resource_cost_usd(
            compute_seconds=compute_seconds,
            cpu_cores=CPU_CORES,
            memory_gib=MEMORY_MIB / 1024,
            rates=MODAL_RATES,
        )
        report = {
            "status": "selected" if selected else "explicit_null",
            "training_examples": len(selections),
            "coarse_layers": list(COARSE_LAYERS),
            "strongest_coarse_candidates": strongest_coarse,
            "refinement_layers": list(fine_layers),
            "ranked_layers": [row.as_dict() for row in ranked],
            "selected_layer": selected_layer,
            "selected_layer_rationale": rationale,
        }
        manifest = {
            "schema_version": 1,
            "configuration_hash": run_hash,
            "configuration": run_config,
            "selected_intervention_contract": {
                "layer_index": selected_layer,
                "token_position_semantics": TOKEN_POSITION,
                "selection_rule": (
                    "matched mean restoration and discrete restoration must exceed both "
                    f"controls; paired advantage >= {MIN_PAIRED_ADVANTAGE}"
                ),
            },
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "render_version": RENDER_VERSION,
                "training_only": True,
            },
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "hardware": {
                "modal_gpu_request": GPU,
                "cuda_device": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
            },
            "timing_seconds": {"measured_compute": round(compute_seconds, 6)},
            "cost": {
                "estimated_resource_cost_usd": cost,
                "credits_included": False,
                "rate_snapshot_date": "2026-06-19",
            },
        }
        saved_manifest, saved_report, created = load_or_create_json_pair(
            run_dir / "manifest.json", manifest, run_dir / "report.json", report
        )
        if created:
            volume.commit()
        return {
            "configuration_hash": run_hash,
            "volume_path": str(run_dir),
            "manifest": saved_manifest,
            "report": saved_report,
        }


@app.local_entrypoint()
def main() -> None:
    result = CausalLayerSweep().run.remote()
    print(json.dumps(result, indent=2))
