"""Modal-only Gemma 4 smoke experiment.

Run with: modal run modal_app.py
Model weights, Torch, Transformers, and inference stay inside Modal.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from countersteer.provenance import (
    CostRates,
    SmokeConfig,
    build_manifest,
    canonical_config_hash,
    estimate_resource_cost_usd,
)


APP_NAME = "countersteer-gemma4-smoke"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fee6332c1abaafb77f6f9624236c63aa2f1d0187"
GPU = "L4"
CPU_CORES = 2.0
MEMORY_MIB = 32_768
VOLUME_PATH = Path("/countersteer")
DEFAULT_PROMPT = (
    "Which statement is mathematically correct? A) 2 + 2 = 4. "
    "B) 2 + 2 = 5. Reply with only A or B."
)
SYSTEM_PROMPT = "You answer carefully and concisely."

# Modal's published base rates on 2026-06-19. Credits are applied by Modal
# afterward, so the manifest deliberately reports a pre-credit estimate.
MODAL_RATES = CostRates(
    gpu_per_second=0.000222,
    cpu_core_per_second=0.0000131,
    memory_gib_per_second=0.00000222,
)

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
class GemmaSmoke:
    @modal.enter()
    def load(self) -> None:
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ["HF_HOME"] = str(VOLUME_PATH / "hf-cache")
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
        volume.commit()

    @modal.method()
    def run(self, prompt: str = DEFAULT_PROMPT, seed: int = 42) -> dict:
        import platform

        import torch

        config = SmokeConfig(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
            seed=seed,
            max_new_tokens=8,
            enable_thinking=False,
            gpu=GPU,
        )
        config_hash = canonical_config_hash(config)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
            )
        inference_seconds = time.perf_counter() - started
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        compute_seconds = self.load_seconds + inference_seconds
        estimated_cost = estimate_resource_cost_usd(
            compute_seconds=compute_seconds,
            cpu_cores=CPU_CORES,
            memory_gib=MEMORY_MIB / 1024,
            rates=MODAL_RATES,
        )
        manifest = build_manifest(
            config=config,
            hardware={
                "modal_gpu_request": GPU,
                "cuda_device": torch.cuda.get_device_name(0),
                "cpu_cores_requested": CPU_CORES,
                "memory_mib_requested": MEMORY_MIB,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
            },
            timing={
                "model_load": round(self.load_seconds, 6),
                "inference": round(inference_seconds, 6),
                "measured_compute": round(compute_seconds, 6),
            },
            cost={
                "estimated_resource_cost_usd": estimated_cost,
                "basis": "measured compute seconds multiplied by published rates",
                "credits_included": False,
                "rate_snapshot_date": "2026-06-19",
                "rates_usd_per_second": {
                    "gpu": MODAL_RATES.gpu_per_second,
                    "cpu_core": MODAL_RATES.cpu_core_per_second,
                    "memory_gib": MODAL_RATES.memory_gib_per_second,
                },
            },
        )
        record = {"manifest": manifest, "result": {"response": response}}

        result_dir = VOLUME_PATH / "results" / "smoke"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{config_hash}.json"
        result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        volume.commit()

        return {
            "response": response,
            "configuration_hash": config_hash,
            "model_revision": MODEL_REVISION,
            "measured_compute_seconds": round(compute_seconds, 6),
            "estimated_resource_cost_usd": estimated_cost,
            "volume_result_path": str(result_path),
            "manifest": manifest,
        }


@app.local_entrypoint()
def main(prompt: str = DEFAULT_PROMPT, seed: int = 42) -> None:
    started = time.perf_counter()
    result = GemmaSmoke().run.remote(prompt=prompt, seed=seed)
    result["client_wall_seconds"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2))
