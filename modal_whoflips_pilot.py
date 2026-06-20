"""Modal-only WhoFlips baseline and prompt-reframing pilot for Gemma 4 E4B."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import modal

from countersteer.artifacts import load_or_create_json_pair
from countersteer.corpus import canonical_hash
from countersteer.provenance import CostRates, estimate_resource_cost_usd
from countersteer.whoflips import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    LABELS,
    RENDER_VERSION,
    SPLIT_VERSION,
    answer_flip_metrics,
    build_whoflips_bundle,
    render_challenge,
    render_question,
)


APP_NAME = "countersteer-whoflips-pilot"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
GPU = "L4"
CPU_CORES = 2.0
MEMORY_MIB = 32_768
VOLUME_PATH = Path("/countersteer")
SEED = 42
TRAIN_PARTITION_SIZE = 600
DEVELOPMENT_PARTITION_SIZE = 200
CONFIRMATION_PARTITION_SIZE = 200
PILOT_TRAIN_SIZE = 200
PILOT_DEVELOPMENT_SIZE = 100
PILOT_SELECTION_VERSION = "whoflips-pilot-hash-selection-v1"
MIN_PROBE_FLIPS = 30
MIN_PROBE_HOLDS = 30
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


def _pilot_ids(ids: tuple[str, ...], size: int, partition: str) -> tuple[str, ...]:
    ranked = sorted(
        ids,
        key=lambda item_id: sha256(
            f"{PILOT_SELECTION_VERSION}:{partition}:{item_id}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < size:
        raise ValueError(f"{partition} pilot needs {size} ids, found {len(ranked)}")
    return tuple(sorted(ranked[:size]))


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
class WhoFlipsPilot:
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
        encoded = {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in LABELS
        }
        if any(len(token_ids) != 1 for token_ids in encoded.values()):
            raise RuntimeError(f"A/B/C/D labels must tokenize once, got {encoded}")
        self.label_ids = {label: token_ids[0] for label, token_ids in encoded.items()}
        volume.commit()

    def _score_messages(self, messages: list[dict], correct_label: str) -> dict:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
        pair = torch.stack(
            [logits[:, self.label_ids[label]] for label in LABELS], dim=-1
        ).float()[0]
        probabilities = torch.softmax(pair, dim=-1)
        predicted_index = int(pair.argmax())
        correct_index = LABELS.index(correct_label)
        strongest_wrong = torch.cat(
            (pair[:correct_index], pair[correct_index + 1 :])
        ).max()
        return {
            "label": LABELS[predicted_index],
            "probability_correct": float(probabilities[correct_index]),
            "correct_logit_margin": float(pair[correct_index] - strongest_wrong),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
        }

    def _evaluate_item(self, item, partition: str) -> list[dict]:
        initial_messages = [{"role": "user", "content": render_question(item)}]
        initial = self._score_messages(initial_messages, item.correct_label)
        records = []
        for strategy, independent_solve in (
            ("ordinary_challenge", False),
            ("independent_solve_prompt", True),
        ):
            messages = [
                *initial_messages,
                {"role": "assistant", "content": initial["label"]},
                {
                    "role": "user",
                    "content": render_challenge(
                        item, independent_solve=independent_solve
                    ),
                },
            ]
            challenged = self._score_messages(messages, item.correct_label)
            records.append(
                {
                    "item_id": item.item_id,
                    "partition": partition,
                    "subject": item.subject,
                    "strategy": strategy,
                    "correct_label": item.correct_label,
                    "initial_label": initial["label"],
                    "challenged_label": challenged["label"],
                    "initial_probability_correct": initial["probability_correct"],
                    "challenged_probability_correct": challenged["probability_correct"],
                    "initial_correct_logit_margin": initial["correct_logit_margin"],
                    "challenged_correct_logit_margin": challenged["correct_logit_margin"],
                    "question_prompt_tokens": initial["prompt_tokens"],
                    "challenge_prompt_tokens": challenged["prompt_tokens"],
                    "source_flip_score": item.source_flip_score,
                    "coercion_model": item.coercion_model,
                }
            )
        return records

    def _load_bundle(self):
        import os

        from datasets import load_dataset

        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
            token=os.environ["HF_TOKEN"],
        )
        if len(dataset) != 2052:
            raise RuntimeError(f"expected 2052 MAXFLIP rows, found {len(dataset)}")
        return build_whoflips_bundle(
            (dict(row) for row in dataset),
            train_size=TRAIN_PARTITION_SIZE,
            development_size=DEVELOPMENT_PARTITION_SIZE,
            confirmation_size=CONFIRMATION_PARTITION_SIZE,
        )

    @modal.method()
    def run(self) -> dict:
        import platform

        import torch

        experiment_started = time.perf_counter()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        bundle = self._load_bundle()
        train_ids = _pilot_ids(bundle.train_ids, PILOT_TRAIN_SIZE, "train")
        development_ids = _pilot_ids(
            bundle.development_ids, PILOT_DEVELOPMENT_SIZE, "development"
        )
        evaluated_ids = set(train_ids) | set(development_ids)
        if evaluated_ids & set(bundle.confirmation_ids):
            raise RuntimeError("pilot attempted to access sealed confirmation ids")
        item_map = {item.item_id: item for item in bundle.items}
        records = []
        for partition, ids in (
            ("train", train_ids),
            ("development", development_ids),
        ):
            for item_id in ids:
                records.extend(self._evaluate_item(item_map[item_id], partition))

        reports = {}
        for partition in ("train", "development"):
            reports[partition] = {}
            for strategy in ("ordinary_challenge", "independent_solve_prompt"):
                selected = [
                    row
                    for row in records
                    if row["partition"] == partition and row["strategy"] == strategy
                ]
                reports[partition][strategy] = answer_flip_metrics(selected)
        ordinary_train = reports["train"]["ordinary_challenge"]
        probe_ready = (
            ordinary_train["flips"] >= MIN_PROBE_FLIPS
            and ordinary_train["holds"] >= MIN_PROBE_HOLDS
        )
        run_config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
            "dataset_hash": bundle.dataset_hash,
            "split_hash": bundle.split_hash,
            "split_version": SPLIT_VERSION,
            "render_version": RENDER_VERSION,
            "pilot_selection_version": PILOT_SELECTION_VERSION,
            "pilot_train_ids": list(train_ids),
            "pilot_development_ids": list(development_ids),
            "confirmation_evaluated": False,
            "minimum_probe_flips": MIN_PROBE_FLIPS,
            "minimum_probe_holds": MIN_PROBE_HOLDS,
            "seed": SEED,
        }
        run_hash = canonical_hash(run_config)
        run_dir = VOLUME_PATH / "results" / "whoflips-pilot" / run_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        records_path = run_dir / "records.json"
        if records_path.exists():
            saved_records = json.loads(records_path.read_text("utf-8"))
            if saved_records != records:
                raise RuntimeError("write-once WhoFlips records differ from rerun")
        else:
            records_path.write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            volume.commit()

        compute_seconds = self.load_seconds + (time.perf_counter() - experiment_started)
        estimated_cost = estimate_resource_cost_usd(
            compute_seconds=compute_seconds,
            cpu_cores=CPU_CORES,
            memory_gib=MEMORY_MIB / 1024,
            rates=MODAL_RATES,
        )
        report = {
            "status": "probe_ready" if probe_ready else "insufficient_probe_labels",
            "probe_readiness": {
                "ready": probe_ready,
                "required_train_flips": MIN_PROBE_FLIPS,
                "required_train_holds": MIN_PROBE_HOLDS,
                "observed_train_flips": ordinary_train["flips"],
                "observed_train_holds": ordinary_train["holds"],
            },
            "partitions": reports,
            "independent_solve_afr_change": {
                partition: (
                    reports[partition]["independent_solve_prompt"]["answer_flip_rate"]
                    - reports[partition]["ordinary_challenge"]["answer_flip_rate"]
                )
                for partition in ("train", "development")
            },
        }
        manifest = {
            "schema_version": 1,
            "configuration_hash": run_hash,
            "configuration": run_config,
            "dataset": {
                "id": DATASET_ID,
                "config": DATASET_CONFIG,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
                "raw_rows": 2052,
                "usable_rows": len(bundle.items),
                "rejected_rows": bundle.rejected_rows,
                "train_partition_size": len(bundle.train_ids),
                "development_partition_size": len(bundle.development_ids),
                "sealed_confirmation_partition_size": len(bundle.confirmation_ids),
                "confirmation_evaluated": False,
                "question_disjoint": True,
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
                "estimated_resource_cost_usd": estimated_cost,
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
    result = WhoFlipsPilot().run.remote()
    print(json.dumps(result, indent=2))
