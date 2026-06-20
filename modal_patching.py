"""PressureBench matched activation patching and causal controls on Modal."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean
import time

import modal

from countersteer.artifacts import load_or_create_json_pair
from countersteer.corpus import canonical_hash
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


APP_NAME = "countersteer-pressurebench-patching"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
GPU = "L4"
CPU_CORES = 2.0
MEMORY_MIB = 32_768
VOLUME_PATH = Path("/countersteer")
LAYER_INDEX = 21
SEED = 42
EXAMPLE_LIMIT = 3
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


def _mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(fmean(present), 6) if present else None


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
class PressureBenchPatching:
    @modal.enter()
    def load(self) -> None:
        import os

        os.environ["HF_HOME"] = str(VOLUME_PATH / "hf-cache")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
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
            language_model = getattr(core, "language_model", None)
            self.layers = getattr(language_model, "layers", None)
        if self.layers is None or len(self.layers) != 42:
            raise RuntimeError("expected 42 Gemma 4 decoder layers")
        label_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in ("A", "B")
        }
        if any(len(ids) != 1 for ids in label_ids.values()):
            raise RuntimeError(f"A/B labels must tokenize once, got {label_ids}")
        self.label_ids = {label: ids[0] for label, ids in label_ids.items()}
        volume.commit()

    def _score(
        self,
        row: dict,
        *,
        capture: bool = False,
        replacement=None,
        patch_enabled: bool = True,
    ):
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        token_index = last_non_padding_index(inputs["attention_mask"][0].tolist())
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if capture:
                captured["activation"] = hidden[:, token_index, :].detach().clone()
            if replacement is None or not patch_enabled:
                return None
            patched = hidden.clone()
            patched[:, token_index, :] = replacement.to(hidden.device, hidden.dtype)
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        should_hook = capture or (replacement is not None and patch_enabled)
        handle = self.layers[LAYER_INDEX].register_forward_hook(hook) if should_hook else None
        try:
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()

        pair = torch.stack(
            [logits[:, self.label_ids["A"]], logits[:, self.label_ids["B"]]],
            dim=-1,
        ).float()[0]
        probabilities = torch.softmax(pair, dim=-1)
        correct_index = 0 if row["correct_label"] == "A" else 1
        wrong_index = 1 - correct_index
        score = {
            "predicted_label": "A" if pair[0] >= pair[1] else "B",
            "logit_a": round(float(pair[0]), 6),
            "logit_b": round(float(pair[1]), 6),
            "probability_a": round(float(probabilities[0]), 8),
            "probability_b": round(float(probabilities[1]), 8),
            "correct_logit_margin": round(
                float(pair[correct_index] - pair[wrong_index]), 6
            ),
            "token_index": token_index,
        }
        return score, captured.get("activation")

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
        pressure_rows = [
            dict(row) for row in dataset if row.get("question_syco")
        ]
        if len(pressure_rows) != 40:
            raise RuntimeError(
                f"expected 40 expert-pressure rows, found {len(pressure_rows)}"
            )
        return build_pressure_bundle(pressure_rows)

    def _run_baseline(self, bundle) -> tuple[list[dict], str]:
        baseline_config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_hash": bundle.dataset_hash,
            "split_hash": bundle.split_hash,
            "conditions": ["neutral", "wrong_belief"],
            "scoring": "constrained-next-token-A-vs-B-v1",
        }
        baseline_hash = canonical_hash(baseline_config)
        baseline_dir = VOLUME_PATH / "results" / "pressurebench-baseline" / baseline_hash
        item_dir = baseline_dir / "items"
        item_dir.mkdir(parents=True, exist_ok=True)
        records = []
        created = 0
        for row in bundle.prompts:
            if row.condition == "correct_belief":
                continue
            prompt = asdict(row)
            path = item_dir / f"{row.item_id}.{row.condition}.json"
            if path.exists():
                record = json.loads(path.read_text("utf-8"))
            else:
                score, _ = self._score(prompt)
                record = {
                    "schema_version": 1,
                    "configuration_hash": baseline_hash,
                    "dataset_hash": bundle.dataset_hash,
                    **prompt,
                    **score,
                }
                path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                created += 1
                if created % 20 == 0:
                    volume.commit()
            records.append(record)
        if created % 20:
            volume.commit()
        baseline_summary = {
            "schema_version": 1,
            "configuration_hash": baseline_hash,
            "configuration": baseline_config,
            "records": len(records),
            "records_created_this_invocation": created,
        }
        summary_path = baseline_dir / "summary.json"
        if not summary_path.exists():
            summary_path.write_text(
                json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8"
            )
            volume.commit()
        return records, baseline_hash

    @modal.method()
    def run(self) -> dict:
        import platform

        import torch

        experiment_started = time.perf_counter()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        bundle = self._load_bundle()
        baseline_records, baseline_hash = self._run_baseline(bundle)
        selections, eligible_count = select_patch_examples(
            baseline_records, bundle.train_ids, EXAMPLE_LIMIT
        )
        if not selections:
            raise RuntimeError("PressureBench produced no usable patching examples")

        prompt_map = {
            (row.item_id, row.condition): asdict(row) for row in bundle.prompts
        }
        run_config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_hash": bundle.dataset_hash,
            "split_hash": bundle.split_hash,
            "source_baseline_hash": baseline_hash,
            "selected_item_ids": [row.item_id for row in selections],
            "behavioral_eligible_count": eligible_count,
            "selection_mode": selections[0].selection_mode,
            "layer_index": LAYER_INDEX,
            "layer_count": len(self.layers),
            "token_position_semantics": TOKEN_POSITION,
            "seed": SEED,
        }
        run_hash = canonical_hash(run_config)
        run_dir = VOLUME_PATH / "results" / "matched-patching" / run_hash
        example_dir = run_dir / "examples"
        example_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for index, selection in enumerate(selections):
            item_id = selection.item_id
            artifact_path = example_dir / f"{item_id}.json"
            if artifact_path.exists():
                results.append(json.loads(artifact_path.read_text("utf-8")))
                continue
            neutral_row = prompt_map[(item_id, "neutral")]
            correct_row = prompt_map[(item_id, "correct_belief")]
            wrong_row = prompt_map[(item_id, "wrong_belief")]
            unrelated_id = next(
                candidate for candidate in bundle.train_ids if candidate != item_id
            )
            unrelated_row = prompt_map[(unrelated_id, "neutral")]

            neutral_score, neutral_activation = self._score(neutral_row, capture=True)
            wrong_score, wrong_activation = self._score(wrong_row, capture=True)
            correct_score, correct_activation = self._score(correct_row, capture=True)
            _, unrelated_activation = self._score(unrelated_row, capture=True)
            if any(
                value is None
                for value in (
                    neutral_activation,
                    wrong_activation,
                    correct_activation,
                    unrelated_activation,
                )
            ):
                raise RuntimeError("activation capture hook did not execute")

            random_seed = SEED + index
            random_replacement, matched_norm, random_norm = self._random_replacement(
                wrong_activation, neutral_activation, random_seed
            )
            scored = {
                "no_patch": wrong_score,
                "disabled_patch": self._score(
                    wrong_row, replacement=neutral_activation, patch_enabled=False
                )[0],
                "matched_neutral": self._score(
                    wrong_row, replacement=neutral_activation
                )[0],
                "correct_user": self._score(
                    wrong_row, replacement=correct_activation
                )[0],
                "unrelated_neutral": self._score(
                    wrong_row, replacement=unrelated_activation
                )[0],
                "random_norm_matched": self._score(
                    wrong_row, replacement=random_replacement
                )[0],
            }
            controls = {}
            for control, score in scored.items():
                controls[control] = {
                    **score,
                    "margin_restoration": round(
                        score["correct_logit_margin"]
                        - wrong_score["correct_logit_margin"],
                        6,
                    ),
                    "restoration_fraction": restoration_fraction(
                        neutral_margin=neutral_score["correct_logit_margin"],
                        wrong_margin=wrong_score["correct_logit_margin"],
                        patched_margin=score["correct_logit_margin"],
                    ),
                    "answer_restored": (
                        score["predicted_label"] == wrong_row["correct_label"]
                        if selection.selection_mode == "behavioral_capitulation"
                        else None
                    ),
                }
            result = {
                "schema_version": 1,
                "configuration_hash": run_hash,
                "selection": asdict(selection),
                "correct_label": wrong_row["correct_label"],
                "wrong_label": wrong_row["wrong_label"],
                "neutral_score": neutral_score,
                "correct_user_score": correct_score,
                "layer_index": LAYER_INDEX,
                "token_position_semantics": TOKEN_POSITION,
                "controls": controls,
                "random_control": {
                    "seed": random_seed,
                    "matched_delta_norm": round(matched_norm, 6),
                    "random_delta_norm": round(random_norm, 6),
                    "absolute_norm_error": round(abs(matched_norm - random_norm), 8),
                },
                "unrelated_item_id": unrelated_id,
            }
            artifact_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            volume.commit()
            results.append(result)

        control_names = list(results[0]["controls"])
        report_controls = {
            control: {
                "mean_margin_restoration": _mean(
                    row["controls"][control]["margin_restoration"] for row in results
                ),
                "mean_restoration_fraction": _mean(
                    row["controls"][control]["restoration_fraction"] for row in results
                ),
                "answer_restoration_rate": _mean(
                    row["controls"][control]["answer_restored"] for row in results
                ),
            }
            for control in control_names
        }
        patching_seconds = time.perf_counter() - experiment_started
        compute_seconds = self.load_seconds + patching_seconds
        cost = estimate_resource_cost_usd(
            compute_seconds=compute_seconds,
            cpu_cores=CPU_CORES,
            memory_gib=MEMORY_MIB / 1024,
            rates=MODAL_RATES,
        )
        report = {
            "status": (
                "behavioral_capitulation_patching"
                if eligible_count
                else "empirical_null_margin_only_diagnostic"
            ),
            "behavioral_eligible_count": eligible_count,
            "selection_mode": selections[0].selection_mode,
            "selected_examples": [row.item_id for row in selections],
            "controls": report_controls,
        }
        manifest = {
            "schema_version": 1,
            "configuration_hash": run_hash,
            "configuration": run_config,
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "render_version": RENDER_VERSION,
                "rows": len(bundle.train_ids) + len(bundle.test_ids),
                "train_items": len(bundle.train_ids),
                "test_items": len(bundle.test_ids),
            },
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "hardware": {
                "modal_gpu_request": GPU,
                "cuda_device": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
                "torch": str(torch.__version__),
            },
            "timing_seconds": {
                "model_load": round(self.load_seconds, 6),
                "baseline_and_patching": round(patching_seconds, 6),
                "measured_compute": round(compute_seconds, 6),
            },
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
            "report": saved_report,
            "manifest": saved_manifest,
            "examples": results,
        }


@app.local_entrypoint()
def main() -> None:
    result = PressureBenchPatching().run.remote()
    print(json.dumps(result, indent=2))
