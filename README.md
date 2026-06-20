# CounterSteer

Counterfactual anti-sycophancy activation steering across languages.

CounterSteer studies whether changing only a user's stated belief causally changes a language model's factual answer, identifies activation layers mediating that effect through matched counterfactual patching, and derives a steering vector intended to reduce capitulation without damaging factual accuracy or genuine reasoning.

## Research objective

The primary experiment uses English, Hindi, and Vietnamese matched prompts on the instruction-tuned `google/gemma-4-E4B-it` model. It compares no intervention, prompting, ordinary contrastive steering, counterfactually filtered steering, random controls, and cross-language vector transfer.

The project will report behavioral metrics, causal patching results, reasoning-retention checks, self-repair traces, and a narrowly scoped Lean certificate over exported empirical activation measurements.

[Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) has 8B total parameters (4.5B effective), a 128K context window, and multilingual support appropriate for the cross-language experiment. The exact model revision will be pinned in every experiment manifest.

## Status

Planning and repository setup for the Apart Global South AI Safety Hackathon 2026.

## Security

Local service credentials belong in `.env`, which is ignored by Git. Use `.env.example` as the variable-name template.

## Modal smoke experiment

The first reproducible path pins Gemma 4 to revision
`fee6332c1abaafb77f6f9624236c63aa2f1d0187`. Model weights and inference remain
on Modal; the local command only submits a request and prints the summary.

```bash
uv sync
uv run modal secret create countersteer-huggingface HF_TOKEN="$HF_TOKEN"
uv run modal run modal_app.py
```

The remote result and provenance manifest are stored in the persistent
`countersteer-data` Volume under `/countersteer/results/smoke/`. The manifest
records the model revision, seed, hardware, timing, configuration hash, and a
transparent pre-credit resource-cost estimate.

## Frozen English baseline

The versioned English corpus contains 60 source items rendered into 180 matched
prompts. Run the resumable forced-choice baseline with:

```bash
uv run modal run modal_baseline.py
```

Per-prompt artifacts, the aggregate report, and its manifest are stored under
`/countersteer/results/english-baseline/<configuration-hash>/` in the same
persistent Modal Volume.

## PressureBench matched activation patching

The patching tracer uses the pinned, CC BY 4.0
[`15juneee/pressure-bench-questions-v1`](https://huggingface.co/datasets/15juneee/pressure-bench-questions-v1)
snapshot. It compares direct GPQA questions with authoritative expert-pressure
prompts that assert the wrong option. Run the fixed mid-layer diagnostic with:

```bash
uv run modal run modal_patching.py
```

The diagnostic uses layer-output residuals at the final non-padding prompt
token and includes no-patch, disabled, matched-neutral, correct-user,
unrelated-neutral, and seeded norm-matched random controls.

Run the train-only coarse-to-fine causal layer locator with:

```bash
uv run modal run modal_layer_sweep.py
```

It evaluates eight regularly spaced layers, refines around the two strongest
coarse candidates, and ranks layers using matched restoration minus the
stronger paired unrelated or norm-matched-random control. The final manifest
stores either the selected layer contract or an explicit null selection.

The pinned training sweep selected decoder layer 41 as the provisional
intervention layer: matched patches restored 10/10 answers versus 8/10 for
unrelated patches and 0/10 for norm-matched random patches. Its conservative
paired restoration advantage was 0.087 with a 95% bootstrap interval of
[-0.091, 0.251], so held-out steering must treat this as a candidate rather
than a statistically settled mechanism.

Run the train-tuned, frozen held-out English steering comparison with:

```bash
uv run modal run modal_steering.py
```

This builds a label-balanced causal vector only from training examples whose
matched layer-41 patch restored the answer. It tunes coefficients on training
data under a five-point neutral-accuracy constraint, then compares causal,
ordinary unfiltered, system-prompt, random, opposite-sign, and no-steering
conditions on the item-disjoint test split.

The frozen 12-item held-out run did not meet the primary steering criterion.
The causal vector raised neutral accuracy from 41.7% to 75.0% but increased
capitulation from 80% to 100%. The system-prompt baseline reduced capitulation
to 60% and reduced counterfactual sensitivity, but lowered neutral accuracy to
33.3% and correct-belief agreement to 75%. These outcomes remain immutable;
the test set was not reused for method selection.

## WhoFlips recovery experiment

The failed residual-stream intervention is not carried directly into the
multilingual phase. Recovery begins with the pinned, CC BY 4.0
[`nafisehNik/WhoFlips`](https://huggingface.co/datasets/nafisehNik/WhoFlips)
MAXFLIP configuration. Its 2,052 unique MMLU questions are filtered before any
Gemma outcomes are observed and deterministically assigned to 600 training,
200 development, 200 sealed confirmation, and reserve question-ID partitions.

Run the Modal-only behavioral pilot with:

```bash
uv run modal run modal_whoflips_pilot.py
```

The pilot evaluates 200 training and 100 development questions. It measures
initial accuracy, conditional Answer Flip Rate, and post-challenge accuracy for
the ordinary MAXFLIP challenge and an independent-solve prompt baseline. The
200 confirmation questions are neither evaluated nor written to pilot
artifacts. Head localization proceeds only when the ordinary training sample
contains at least 30 initially-correct flips and 30 initially-correct holds.

The first immutable pilot passed that gate. On 200 training questions, Gemma was
initially correct on 136; the ordinary challenge produced 95 flips and 41
holds. On 100 development questions, ordinary AFR was 62.7% (47/75), while the
independent-solve prompt reduced AFR to 22.7% (17/75), a 40 percentage-point
reduction. The run took 131.0 measured compute-seconds on an NVIDIA L4 with an
estimated pre-credit resource cost of $0.041832. The sealed confirmation
partition was not evaluated. The immutable configuration hash is
`75fd1dba2490f61db670fe0103e5ce839849d1214d2c7e61f8dc788d4a71f887`.
