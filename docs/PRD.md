## Problem Statement

Language models frequently change otherwise-correct factual answers after a user confidently asserts an incorrect belief. Existing anti-sycophancy interventions can improve benchmark behavior while leaving the underlying mechanism intact, introducing indiscriminate contrarianism, or damaging useful reasoning. Ordinary contrastive activation steering is also correlational: a direction that separates sycophantic and non-sycophantic examples is not necessarily a causal mediator of the behavior.

The project needs a small-compute, reproducible way to identify where user-belief pressure causally affects a model, derive an intervention only from causally validated examples, and test whether that intervention removes capitulation without degrading accuracy or genuine reasoning. Because the hackathon is situated in the Global South and the Asia technical-safety track explicitly motivates multilingual sycophancy research, the method must also determine whether an English-derived intervention transfers to Hindi and Vietnamese or whether language-specific directions are required.

The result must be scientifically useful even if steering fails. It must distinguish behavioral success, activation-level changes, reasoning preservation, and machine-checked geometric claims rather than collapsing them into an unsupported claim of universal safety.

## Solution

Build CounterSteer, an experimental pipeline for counterfactual anti-sycophancy activation steering across English, Hindi, and Vietnamese.

CounterSteer will construct matched factual prompts in which only the user's stated belief changes. It will first measure how much an incorrect user belief changes the model's answer. For cases where the model knows the answer under a neutral prompt but capitulates under an incorrect-belief prompt, the system will replace selected internal activations with activations from the matched neutral counterfactual. Layers where the matched patch restores the correct answer, while unrelated and random patches do not, will be treated as candidate causal mediators.

The system will aggregate paired activation differences only from causally successful training examples to produce an anti-sycophancy direction. It will apply that direction at inference time, choose intervention strength using training data under an accuracy-preservation constraint, and evaluate all claims on a frozen item-disjoint test split. It will compare the causal vector against no steering, an anti-sycophancy system prompt, ordinary unfiltered contrastive steering, norm-matched random steering, and opposite-sign steering.

The multilingual experiment will learn English, Hindi, Vietnamese, and shared directions and evaluate every direction in every language. The primary output will be a cross-language transfer matrix reporting capitulation reduction, accuracy retention, counterfactual invariance, and uncertainty estimates.

Reasoning preservation will be evaluated on a small, informative subset with True Thinking Score methodology. Later-layer projection tracking will test whether the removed direction is reconstructed after intervention. A standalone Lean project will verify bounded geometric predicates over quantized, exported empirical traces. The formal artifact will explicitly certify only those finite measurements, not the semantic validity of the direction or universal behavioral safety.

The primary model will be Qwen3-0.6B. The core project performs inference and activation interventions rather than model-weight training. Modal will provide scale-to-zero GPU execution, persistent model and result caching, and parallel experiment shards. Adaption will be used narrowly for multilingual data adaptation and quality review after a quoted pilot; it will not create factual labels or substitute its quality score for project metrics.

## User Stories

1. As an AI-safety researcher, I want matched neutral, correct-belief, and wrong-belief prompts, so that I can isolate the effect of the user's stated belief.
2. As an AI-safety researcher, I want answer labels balanced across items, so that a steering direction cannot exploit a fixed answer-position shortcut.
3. As an AI-safety researcher, I want item-disjoint training and test splits, so that steering performance is evaluated without question leakage.
4. As a multilingual evaluator, I want semantically matched English, Hindi, and Vietnamese prompts, so that cross-language results are comparable.
5. As a multilingual evaluator, I want translation provenance and review status recorded, so that synthetic-language limitations are visible.
6. As a dataset maintainer, I want deterministic counterfactual template generation, so that the three prompt conditions differ only in controlled fields.
7. As a dataset maintainer, I want stable item identifiers and dataset hashes, so that every result can be traced to an exact dataset version.
8. As a researcher, I want the unsteered model's neutral accuracy measured first, so that factual ignorance is not mistaken for sycophancy.
9. As a researcher, I want capitulation measured only when the neutral model answers correctly, so that the metric captures deference rather than inability.
10. As a researcher, I want an answer-logit margin in addition to generated answers, so that subtle changes in preference are measurable.
11. As a researcher, I want a counterfactual-sensitivity metric, so that I can quantify how strongly a changed user belief moves the answer distribution.
12. As a researcher, I want an anti-sycophancy system-prompt baseline, so that a complex intervention must beat a simple method.
13. As an interpretability researcher, I want activations captured at controlled token positions, so that layer comparisons are meaningful.
14. As an interpretability researcher, I want matched neutral activations patched into wrong-belief runs, so that I can test causal mediation.
15. As an interpretability researcher, I want a coarse-to-fine layer sweep, so that causal layers can be found without an unnecessarily expensive exhaustive search.
16. As an interpretability researcher, I want unrelated-example patch controls, so that generic activation replacement is not mistaken for a matched counterfactual effect.
17. As an interpretability researcher, I want norm-matched random patch controls, so that restoration cannot be attributed only to perturbation magnitude.
18. As an interpretability researcher, I want opposite-direction controls, so that the predicted sign of the intervention is falsifiable.
19. As a researcher, I want patch restoration rates with uncertainty estimates, so that candidate causal layers are selected from evidence rather than anecdotes.
20. As a steering researcher, I want vector construction restricted to causally successful training examples, so that correlational confounds are reduced.
21. As a steering researcher, I want an ordinary unfiltered contrastive vector baseline, so that the value of counterfactual filtering is directly tested.
22. As a steering researcher, I want coefficient selection performed only on training data, so that test results remain unbiased.
23. As a steering researcher, I want neutral accuracy enforced as a constraint during coefficient selection, so that lower capitulation cannot be purchased by destroying capability.
24. As a steering researcher, I want wrong-belief capitulation reduced on held-out items, so that the intervention demonstrates generalization.
25. As a safety evaluator, I want correct-belief prompts evaluated, so that the intervention does not simply make the model disagree with every user.
26. As a safety evaluator, I want refusal, hedging, answer length, and label-bias checks, so that degenerate anti-sycophancy strategies are detected.
27. As a safety evaluator, I want paired bootstrap confidence intervals over items, so that reported improvements include sampling uncertainty.
28. As a multilingual researcher, I want one steering vector learned per language, so that language-specific mechanisms can be detected.
29. As a multilingual researcher, I want a shared multilingual vector, so that a language-general intervention can be compared against specialized vectors.
30. As a multilingual researcher, I want every learned vector evaluated in every language, so that transfer and interference are visible in one matrix.
31. As a reasoning researcher, I want TTS evaluated on informative successes, failures, and transfer disagreements, so that limited compute targets the most diagnostic cases.
32. As a reasoning researcher, I want TTS retention compared with the unsteered model, so that behavioral improvement is not confused with reasoning preservation.
33. As a researcher, I want TrueThinking-preserving dual steering attempted only when the simple causal vector damages TTS, so that complexity is justified by evidence.
34. As an interpretability researcher, I want the sycophancy-direction projection tracked through later layers, so that activation self-repair can be detected.
35. As an interpretability researcher, I want behavioral outcomes compared with projection self-repair, so that geometric and behavioral recovery are not conflated.
36. As a formal-methods researcher, I want empirical activation measurements quantized reproducibly, so that they can be checked in Lean.
37. As a formal-methods researcher, I want Lean to verify perturbation, projection, reasoning-preservation, and repair bounds, so that arithmetic claims are machine checked.
38. As a formal-methods researcher, I want the certificate to contain no unresolved placeholders, so that compilation represents a completed proof artifact.
39. As a reviewer, I want the certificate's trust boundary stated explicitly, so that finite geometric verification is not marketed as universal safety.
40. As an experiment operator, I want model revision, random seeds, layer, coefficient, vector hash, and dataset hash captured in each run manifest, so that results are reproducible.
41. As an experiment operator, I want failed shards recorded without invalidating successful shards, so that expensive batches can be resumed safely.
42. As an experiment operator, I want cached model weights and write-once experiment artifacts, so that reruns do not repeatedly download or overwrite evidence.
43. As an experiment operator, I want GPU concurrency capped and idle containers scaled to zero, so that the experiment stays within a realistic compute budget.
44. As an experiment operator, I want per-stage cost estimates and stop conditions, so that available credits are not treated as a spending target.
45. As an Adaption user, I want a small quoted pilot before adapting the full dataset, so that credit use is known before execution.
46. As an Adaption user, I want Adaption excluded from factual-label generation, so that the scientific ground truth remains independently controlled.
47. As a demo viewer, I want to select a language and question and compare unsteered and steered answers, so that the intervention is understandable immediately.
48. As a demo viewer, I want to see layer, coefficient, logit margin, and counterfactual sensitivity, so that the visible answer is connected to quantitative evidence.
49. As a demo viewer, I want to see a self-repair trace and certificate status, so that behavioral, activation, and formal evidence are shown separately.
50. As a hackathon judge, I want precomputed demo examples, so that the presentation remains reliable without waiting for a live GPU.
51. As a hackathon judge, I want a baseline comparison table and cross-language heatmap, so that the contribution can be assessed quickly.
52. As a hackathon judge, I want negative findings reported with the same rigor as successes, so that the project remains informative if transfer or certification fails.
53. As a future contributor, I want stable experiment interfaces and documented manifests, so that additional languages and models can be added later.
54. As a repository user, I want credentials stored only in ignored local configuration, so that service tokens never enter source control.

## Implementation Decisions

- The initial repository is a greenfield research codebase; there is no existing implementation or test precedent to preserve.
- The primary model is Qwen3-0.6B. A larger model is a replication stretch goal, not a dependency of the core result.
- The core project uses activation inference and intervention rather than weight training. Full fine-tuning and DPO are not required.
- The primary task is controlled two-option factual answering. Short outputs make answer scoring reliable and keep compute costs low.
- The dataset contains 60 source items across mathematics/reasoning and factual knowledge, expanded into three counterfactual conditions and three languages.
- Items, not rendered prompts, are the unit of train/test splitting and statistical resampling.
- Counterfactual templates are deterministic. Correct answer positions are randomized and balanced.
- Translation provenance, adaptation configuration, and human-review status are dataset metadata.
- Adaption is isolated behind a dataset-adaptation module that accepts source rows and returns versioned candidate translations or paraphrases. It never owns ground-truth answers.
- The model-runtime module provides a stable interface for tokenization, deterministic generation, answer-logit extraction, activation capture, activation intervention, and run metadata.
- The counterfactual-corpus module encapsulates item validation, condition rendering, label balancing, language variants, frozen splitting, and hashing.
- The activation-intervention module encapsulates residual-stream capture and patching at a logical token position, allowing the layer-search algorithm to remain independent of model-hook details.
- The causal-layer locator performs a coarse-to-fine sweep and ranks layers using matched restoration relative to unrelated and random controls.
- The vector-estimation module accepts causally validated paired activations and produces normalized per-language and shared directions with immutable hashes.
- The steering runtime applies a selected direction and coefficient at a selected layer without modifying model weights.
- Coefficients are selected on training items by minimizing wrong-belief capitulation subject to a neutral-accuracy constraint.
- The evaluation module computes neutral accuracy, capitulation, correct-belief agreement, answer-logit margins, counterfactual sensitivity, degeneracy metrics, and paired uncertainty estimates.
- The transfer evaluator treats learned-vector language and evaluation language as separate axes and emits a complete transfer matrix.
- The TTS validator runs on a deliberately small diagnostic subset rather than the full corpus.
- Dual-direction TrueThinking preservation is enabled only if the initial causal vector fails the TTS-retention criterion.
- The self-repair analyzer records projection onto the learned direction at each layer after intervention and aligns those traces with final behavioral outcomes.
- The experiment orchestrator executes immutable, resumable shards keyed by experiment configuration and stores manifests separately from aggregate reports.
- Modal uses a persistent Volume for model cache and experiment artifacts, scale-to-zero execution, base-region pricing, and bounded concurrency. L4 is preferred and A10 is the fallback.
- Expected Modal spend for the primary experiment is approximately 10 to 20 US dollars, with a hard stop around 30 to 35 dollars unless a larger-model replication is explicitly enabled.
- Adaption spend is determined through estimate-only pilot calls because no public dollar-per-row conversion is available. The initial estimate is capped to a small row subset.
- The certificate-export module converts selected measurements to deterministic rational or integer representations and records quantization error.
- Lean verifies finite geometric predicates and generic algebraic lemmas. It does not prove semantic identification, dataset representativeness, or universal behavioral safety.
- The demo defaults to precomputed examples and can optionally invoke a live backend when available.
- The research report separates behavioral evidence, activation evidence, reasoning evidence, and formal evidence.
- Natural Language Autoencoder work, model-weight training, and large multi-model benchmarking are excluded from the critical path.

## Testing Decisions

- Good tests assert observable scientific or user-facing behavior rather than internal implementation details. Tests should remain valid when hook implementation, batching strategy, or storage layout changes.
- The counterfactual-corpus module will be tested for deterministic rendering, balanced labels, item-disjoint splits, schema validation, translation metadata, and stable hashes.
- The model-runtime interface will be tested with a tiny deterministic fixture model where known logits and activations make capture and intervention outcomes predictable.
- Activation patching will be tested for correct layer and token targeting, unchanged behavior when the patch is disabled, and reproducibility with fixed seeds.
- The causal-layer locator will be tested on synthetic interventions with one known causal layer and on null data where no layer should be selected.
- Vector estimation will be tested for paired aggregation, balancing, normalization, causally validated filtering, and rejection of empty or malformed evidence.
- Steering will be tested for sign, coefficient scaling, norm-matched controls, and absence of persistent weight mutation.
- Evaluation will be tested against hand-calculated examples for accuracy, capitulation, counterfactual sensitivity, degeneracy flags, transfer matrices, and bootstrap reproducibility.
- TTS integration will be tested at its boundary using fixed perturbation outcomes rather than asserting internal behavior of the external implementation.
- Self-repair analysis will be tested on constructed projection sequences with known recovery and non-recovery patterns.
- Run manifests will be tested for required provenance fields and stable experiment identifiers.
- Cost-control behavior will be tested by validating maximum row counts, maximum containers, estimate-only Adaption calls, and experiment-level spend guards without launching paid work in unit tests.
- Certificate export will be tested for deterministic quantization, explicit error bounds, stable hashes, and rejection of non-finite values.
- The Lean project will be compiled in continuous integration, and the build will fail if proofs contain unresolved placeholders or generated certificate data is invalid.
- The demo will receive smoke tests for loading precomputed results and presenting separate behavioral, activation, TTS, and certificate states.
- End-to-end tests will use a tiny local fixture corpus and fixture model. Paid remote GPU tests will be opt-in integration tests, not routine unit tests.
- Because the repository is new, there is no internal prior art. Test design follows research-software practice: deterministic fixtures, golden metric summaries where appropriate, property checks for invariants, and explicit separation between local and paid integration tests.

## Out of Scope

- Training a frontier model or claiming that results generalize to frontier systems.
- Full fine-tuning, DPO, RLHF, or mandatory LoRA training.
- More than three languages in the primary experiment.
- More than one primary model before the full core pipeline is complete.
- Natural Language Autoencoder training or making NLA output a required deliverable.
- An unrestricted free-form sycophancy benchmark; free-form examples are limited to qualitative demonstration.
- Proving that the learned vector semantically represents sycophancy in every context.
- Formally verifying the entire transformer or proving universal end-to-end behavioral safety.
- Treating Adaption's data-quality score as a safety, truthfulness, or sycophancy metric.
- Using generated translations without recording their provenance and review status.
- Spending all available Modal or Adaption credits merely because they are available.
- Building a production inference service with uptime, authentication, or multi-tenant guarantees.

## Further Notes

- Primary success means at least a 20-percentage-point reduction in wrong-belief capitulation on the frozen test split, no more than a 5-point decrease in neutral accuracy, reduced counterfactual sensitivity, and improvement beyond random and ordinary contrastive baselines.
- TTS retention of 0.8 is a diagnostic threshold rather than a universal safety constant and will be reported with sensitivity analysis when feasible.
- A failure of English steering to transfer can be the central result if language-matched vectors perform differently.
- A geometric certificate passing while behavioral or TTS checks fail is an important negative result, not a reason to hide the certificate.
- The demo should remain functional from cached results even if Modal or another external service is unavailable during judging.
- All credentials are local-only, ignored by Git, and must never appear in issues, logs, reports, manifests, or committed configuration.
