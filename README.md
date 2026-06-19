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
