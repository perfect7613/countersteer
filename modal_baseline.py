"""Run the frozen 180-prompt English baseline entirely on Modal."""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from countersteer.artifacts import load_or_create_json_pair
from countersteer.corpus import canonical_hash, load_english_bundle
from countersteer.metrics import build_baseline_report
from countersteer.provenance import CostRates, estimate_resource_cost_usd


APP_NAME = "countersteer-english-baseline"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
GPU = "L4"
CPU_CORES = 2.0
MEMORY_MIB = 32_768
VOLUME_PATH = Path("/countersteer")
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 42
ARTIFACT_SCHEMA_VERSION = 2
MODAL_RATES = CostRates(0.000222, 0.0000131, 0.00000222)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("countersteer-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("countersteer-huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.14.0",
        "huggingface-hub==1.20.1",
        "safetensors==0.8.0",
        "torch==2.12.1",
        "transformers==5.12.1",
    )
    .add_local_python_source("countersteer")
)


def _chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


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
class GemmaEnglishBaseline:
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
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        volume.commit()

    def _label_token_ids(self) -> dict[str, int]:
        token_ids = {
            label: self.tokenizer.encode(label, add_special_tokens=False)
            for label in ("A", "B")
        }
        if any(len(ids) != 1 for ids in token_ids.values()):
            raise RuntimeError(f"A/B labels must each tokenize once, got {token_ids}")
        return {label: ids[0] for label, ids in token_ids.items()}

    @modal.method()
    def run(
        self,
        payload: dict,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> dict:
        import platform

        import torch

        if not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be between 1 and 8")
        expected_dataset_hash = canonical_hash(
            {
                "template_version": payload["template_version"],
                "prompts": payload["prompts"],
            }
        )
        expected_split_hash = canonical_hash(
            {
                "version": payload["split_version"],
                "train": payload["train_ids"],
                "test": payload["test_ids"],
            }
        )
        if expected_dataset_hash != payload["dataset_hash"]:
            raise ValueError("dataset hash does not match rendered prompts")
        if expected_split_hash != payload["split_hash"]:
            raise ValueError("split hash does not match frozen split")
        if len(payload["prompts"]) != 180:
            raise ValueError("baseline requires exactly 180 rendered prompts")

        run_config = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_hash": payload["dataset_hash"],
            "split_hash": payload["split_hash"],
            "seed": seed,
            "batch_size": batch_size,
            "scoring": "constrained-next-token-A-vs-B-v1",
            "enable_thinking": False,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        }
        run_hash = canonical_hash(run_config)
        run_dir = VOLUME_PATH / "results" / "english-baseline" / run_hash
        item_dir = run_dir / "items"
        item_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        label_ids = self._label_token_ids()
        records: list[dict] = []
        pending: list[dict] = []
        resumed_records = 0
        for row in payload["prompts"]:
            path = item_dir / f"{row['item_id']}.{row['condition']}.json"
            if path.exists():
                record = json.loads(path.read_text("utf-8"))
                if record["configuration_hash"] != run_hash:
                    raise ValueError(f"stale artifact at {path}")
                records.append(record)
                resumed_records += 1
            else:
                pending.append(row)

        inference_seconds = 0.0
        for batch in _chunks(pending, batch_size):
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch
            ]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            ).to(self.model.device)
            started = time.perf_counter()
            with torch.inference_mode():
                next_logits = self.model(**inputs).logits[:, -1, :]
            inference_seconds += time.perf_counter() - started

            selected = torch.stack(
                [next_logits[:, label_ids["A"]], next_logits[:, label_ids["B"]]],
                dim=-1,
            ).float()
            probabilities = torch.softmax(selected, dim=-1).cpu().tolist()
            label_logits = selected.cpu().tolist()
            for row, logits, probs in zip(batch, label_logits, probabilities):
                correct_index = 0 if row["correct_label"] == "A" else 1
                wrong_index = 1 - correct_index
                record = {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "configuration_hash": run_hash,
                    "dataset_hash": payload["dataset_hash"],
                    "split_hash": payload["split_hash"],
                    **row,
                    "predicted_label": "A" if logits[0] >= logits[1] else "B",
                    "logit_a": round(float(logits[0]), 6),
                    "logit_b": round(float(logits[1]), 6),
                    "probability_a": round(float(probs[0]), 8),
                    "probability_b": round(float(probs[1]), 8),
                    "correct_logit_margin": round(
                        float(logits[correct_index] - logits[wrong_index]), 6
                    ),
                }
                artifact_path = item_dir / f"{row['item_id']}.{row['condition']}.json"
                artifact_path.write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
                records.append(record)
            volume.commit()

        records.sort(key=lambda row: (row["item_id"], row["condition"]))
        report = build_baseline_report(
            records, payload["train_ids"], payload["test_ids"]
        )
        compute_seconds = self.load_seconds + inference_seconds
        estimated_cost = estimate_resource_cost_usd(
            compute_seconds=compute_seconds,
            cpu_cores=CPU_CORES,
            memory_gib=MEMORY_MIB / 1024,
            rates=MODAL_RATES,
        )
        creation_manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "configuration_hash": run_hash,
            "configuration": run_config,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "dataset": {
                "hash": payload["dataset_hash"],
                "split_hash": payload["split_hash"],
                "template_version": payload["template_version"],
                "source_items": 60,
                "rendered_prompts": 180,
            },
            "hardware": {
                "modal_gpu_request": GPU,
                "cuda_device": torch.cuda.get_device_name(0),
                "cpu_cores_requested": CPU_CORES,
                "memory_mib_requested": MEMORY_MIB,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
            },
            "timing_seconds": {
                "model_load": round(self.load_seconds, 6),
                "new_inference": round(inference_seconds, 6),
                "measured_compute": round(compute_seconds, 6),
            },
            "resumption": {
                "records_reused": resumed_records,
                "records_created": len(pending),
            },
            "cost": {
                "estimated_resource_cost_usd": estimated_cost,
                "basis": "measured compute seconds multiplied by published rates",
                "credits_included": False,
                "rate_snapshot_date": "2026-06-19",
            },
        }
        manifest_path = run_dir / "manifest.json"
        report_path = run_dir / "report.json"
        manifest, report, artifacts_created = load_or_create_json_pair(
            manifest_path, creation_manifest, report_path, report
        )
        if artifacts_created:
            volume.commit()

        return {
            "configuration_hash": run_hash,
            "dataset_hash": payload["dataset_hash"],
            "split_hash": payload["split_hash"],
            "records": len(records),
            "records_reused": resumed_records,
            "records_created": len(pending),
            "estimated_resource_cost_usd": estimated_cost,
            "measured_compute_seconds": round(compute_seconds, 6),
            "volume_path": str(run_dir),
            "report": report,
            "manifest": manifest,
            "invocation": {
                "records_reused": resumed_records,
                "records_created": len(pending),
                "measured_compute_seconds": round(compute_seconds, 6),
                "estimated_resource_cost_usd": estimated_cost,
            },
        }


@app.local_entrypoint()
def main(batch_size: int = DEFAULT_BATCH_SIZE, seed: int = DEFAULT_SEED) -> None:
    bundle = load_english_bundle()
    result = GemmaEnglishBaseline().run.remote(
        payload=bundle.remote_payload(), batch_size=batch_size, seed=seed
    )
    print(json.dumps(result, indent=2))
