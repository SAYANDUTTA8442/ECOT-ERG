# ECOT-ERG: Towards Causally Faithful Empathetic Dialogue Generation via Structured Appraisal Reasoning

> **ACL ARR 2026 Submission #12976** · Under Review · Preferred Venue: EMNLP

<p align="center">
  <img src="https://img.shields.io/badge/Status-Under%20Review-yellow" />
  <img src="https://img.shields.io/badge/Venue-EMNLP%202026-blue" />
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-green" />
  <img src="https://img.shields.io/badge/Task-Empathetic%20Dialogue-purple" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" />
</p>

---

## Overview

**ECOT-ERG** (*Emotion-Cause-Oriented Thought for Empathetic Response Generation*) is a structured causal empathy-reasoning framework that decomposes empathetic dialogue generation into interpretable appraisal-theoretic intermediate steps.

Unlike conventional models that treat empathetic generation as a single-step conditional text generation problem, ECOT-ERG reasons explicitly over **what caused** the emotion, **which cause dominates**, **whether the situation is controllable**, and **which empathy strategy to apply** — before generating the final response.

> *"An empathetic response should recognize that the emotionally dominant event is the loss of a parent rather than treating all contributing causes equally important."*

---

## Motivation

Most existing empathetic response generation (ERG) systems are:

| Limitation | Description |
|---|---|
| **Emotion-only grounded** | Match detected emotion labels without reasoning about *why* the speaker feels that way |
| **Cause-unaware** | Do not model which cause is emotionally dominant in multi-cause scenarios |
| **Controllability-blind** | Treat "I missed work because I overslept" and "I missed work because I was hospitalized" as equivalently deserving the same empathetic response |
| **Opaque** | Produce fluent responses with no interpretable reasoning chain |

ECOT-ERG addresses all four limitations through structured appraisal-theoretic supervision grounded in psychological attribution theory (Weiner, 1985) and appraisal theory (Moors, 2017).

---

## Framework

### Reasoning Pipeline

```
Dialogue Context  X = {x₁, x₂, ..., xₙ}
        │
        ▼
① Emotion Identification         →  e  (e.g., sadness)
        │
        ▼
② Emotional Cause Extraction     →  C = {c₁, c₂, ..., cₖ}
        │
        ▼
③ Dominant Cause Selection       →  c* = argmax rᵢ  (causal relevance ranking)
        │
        ▼
④ Controllability Appraisal      →  γ ∈ {0, 1}  (uncontrollable / controllable)
        │
        ▼
⑤ Empathy Strategy Selection     →  s* ∈ {Validation, Reframing, Exploration}
        │
        ▼
⑥ Response Generation            →  Y  (causally grounded empathetic response)
```

The model is trained autoregressively on the concatenated chain: `[e; C; c*; γ; s; Y]` with explicit intermediate supervision at each stage.

---

### Empathy Strategy Taxonomy

| Strategy | Trigger Condition | Example Fragment | Distribution |
|---|---|---|---|
| **Validation** | Uncontrollable causes; grief; loss; externally imposed distress | *"That sounds incredibly painful and your feelings make complete sense."* | 48.8% |
| **Reframing** | Controllable causes with fixable components; latent positive aspects | *"It sounds hard right now, but the choice you made came from wanting the best for yourself."* | 35.1% |
| **Exploration** | Ambiguous or unclear causes; indeterminate controllability | *"What part of this is weighing on you most right now?"* | 16.1% |

Strategy selection follows a formal precedence rule grounded in EPITOME (Jones et al., 2024): **Validation** for uncontrollable (γ = 0), **Reframing** for controllable (γ = 1), **Exploration** as default for indeterminate cases.

---

## Training Tracks

### Track A — Structured Supervision (Small-Scale)
- **Model**: `google/flan-t5-base`
- **Objective**: Full structured chain supervision (CE + BCE + DPO)
- Cross-entropy loss applied uniformly over all reasoning positions
- DPO preference optimization over preferred/rejected reasoning chains

### Track B — Structured Prompting (Frontier LLMs)
- **Model**: GPT-4 (zero-shot access, no parameter tuning)
- Sequential appraisal-theoretic prompting with XML-tagged reasoning stages
- Few-shot exemplars stratified by strategy class, controllability label, and cause category

### Track C — LoRA + DPO Alignment (Mid-Scale)
- **Model**: `meta-llama/Llama-2-7b-chat-hf`
- LoRA adaptation on all linear projection matrices (r=16, α=32)
- Followed by DPO-based causal alignment using same preference pair construction as Track A

---

## Dataset & Supervision Resources

All experiments use the **EmpatheticDialogues** corpus (Rashkin et al., 2019) with standard train/val/test splits.

| Resource | Type | Source | Purpose |
|---|---|---|---|
| EmpatheticDialogues | Existing | Rashkin et al., 2019 | Main dialogue corpus |
| Emotion-Cause Dataset | Derived | Situation field (EmpatheticDialogues) | Cause generation supervision |
| Causal Relevance Dataset | New | Generated candidate causes | Relevance ranking supervision |
| Preference Pair Dataset | New | Structured reasoning chains | DPO alignment |
| Controllability Dataset | New | Manually annotated subset (500 instances) | Controllability classification |
| EPITOME | Existing | Jones et al., 2024 | Empathy strategy supervision |
| ECOT Structured Corpus | New | Integrated reasoning chains | End-to-end ECOT training |

**Training/Validation split**: 7,780 training / 865 validation instances with full reasoning chain annotations.

---

## Controllability Annotation Pipeline

Binary controllability labels `γ ∈ {0, 1}` are constructed through a two-stage pipeline:

1. **Manual Seed Annotation**: 500 dialogues annotated via multi-annotator voting, adjudicated by a domain expert (κ = 0.828, raw agreement 92%)
2. **Classifier-Based Propagation**: RoBERTa-base classifier trained on seed data; high-confidence filtering at threshold ≥ 0.75 expands coverage to ~2,000 instances

**Final dataset distribution**: ~64% controllable, ~36% uncontrollable — consistent with the manually annotated subset.

**Label noise robustness**: System accuracy remains ≥ 0.65 and Class-1 F1 ≥ 0.71 under up to 20% injected noise; substantial degradation only above 40%.

---

## DPO Preference Pair Construction

Each preference pair `(z⁺, z⁻)` is constructed deterministically (no human annotation or model-based generation):

| Pair Type | Construction Method |
|---|---|
| **Hard negatives** | Cross-conversation cause substitution within the same emotion group; filtered on conversation ID and 40-character prefix overlap |
| **Easy negatives** | Cause drawn from opposite polarity emotion group (positive→negative or negative→positive) |

Negative chains carry compound errors: incorrect cause substitution + controllability reversal + strategy mismatch.

---

## Results

### Main Evaluation (Table 3)

| Model | BERTScore | EASE | Cause F1 | Rel. Acc. | PPL↓ | D-1 | D-2 |
|---|---|---|---|---|---|---|---|
| MoEL | 0.312 | -0.1671 | — | — | 35.02 | 0.010 | 0.682 |
| MIME | 0.367 | -0.2338 | — | — | 37.24 | 0.508 | 0.664 |
| CEM | 0.490 | 0.0125 | — | — | 36.33 | 0.622 | 0.839 |
| CFEG | 0.652 | 0.1101 | 0.6218 | 0.527 | 6.67 | 0.926 | 0.952 |
| B1 — Tuned T5 | 0.693 | 0.0973 | 0.0018 | 0.462 | 11.72 | 0.523 | 0.901 |
| **ECOT-T5 (Track A)** | **0.837** | **0.1389** | **0.7662** | **0.596** | **5.89** | **0.957** | **0.961** |
| **ECOT-LLaMA (Track C)** | **0.844** | **0.1641** | **0.7812** | **0.581** | **6.26** | **0.927** | **0.551** |
| **ECOT-LLM (Track B)** | **0.854** | **0.1552** | **0.8093** | **0.586** | **5.01** | **0.981** | **0.988** |

### Ablation Study (Track A)

| Model Variant | BERTScore | Cause F1 | Rel. Acc. | Key Finding |
|---|---|---|---|---|
| ECOT-full | 0.8372 | 0.7662 | 0.5969 | — |
| ECOT-noStrategy | 0.6946 | 0.7662 | 0.3128 | ROUGE-2 drops >50% |
| ECOT-noCausePriority | 0.6612 | 0.1209 | 0.3726 | Catastrophic Cause F1 collapse |
| ECOT-noControllability | 0.6943 | 0.7662 | 0.2189 | Rel. Acc. at lowest point |

### Scale & Alignment Analysis (LLaMA Family)

| Model | BERTScore-F1 | Δ vs Base |
|---|---|---|
| Base LLaMA-2-7B | 0.6593 | — |
| LLaMA-2-7B + LoRA only | 0.8293 | +0.1700 |
| **LLaMA-2-7B + LoRA + DPO** | **0.8446** | **+0.1853** |
| TinyLlama-1.1B | 0.7602 | +0.1009 |
| LLaMA-2-13B (Base, untuned) | 0.8135 | +0.1542 |

> Our alignment-tuned 7B variant outperforms the raw untuned 13B model.

### Human Evaluation (3 independent annotators per instance)

| Dimension | ECOT | Baselines | Δ | Win% | Cohen's d |
|---|---|---|---|---|---|
| Empathy | 4.05 ± 0.52 | 3.17 ± 0.56 | +0.875 | 86.7% | 1.41 |
| Coherence | 4.03 ± 0.48 | 3.21 ± 0.54 | +0.822 | 88.3% | 1.45 |
| Helpfulness | 4.04 ± 0.46 | 3.18 ± 0.59 | +0.861 | 89.2% | 1.40 |
| Emotional Appropriateness | 4.08 ± 0.42 | 3.22 ± 0.60 | +0.864 | 90.0% | 1.53 |
| Reasoning Clarity | 4.01 ± 0.46 | 3.12 ± 0.57 | +0.889 | 87.5% | 1.45 |

All differences significant at p < 0.0001. A/B preference study: ECOT-T5 preferred in **Cause Understanding** at 60.0% vs 16.7% for baseline.

---

## Qualitative Examples

### Example 1 — Controllable / Negative Valence

> **Utterance**: "My boss gives us a huge amount of work but wants to sit and chat the whole time we're trying to focus."

| System | Response | BERTScore | EASE |
|---|---|---|---|
| **B1 (Baseline)** | *"I'm really sorry you're going through this. You're not alone in this."* | 0.626 | 0.087 |
| **ECOT-T5** | *"That sounds genuinely frustrating — it's hard to concentrate when you're being pulled in different directions. What have you tried so far?"* | 0.702 | 0.291 |

Strategy applied: **Exploration** (controllable, indeterminate intent)

### Example 2 — Uncontrollable / Deep Negative Valence

> **Utterance**: "I found my dog passed away when I came home today."

| System | Response | BERTScore | EASE |
|---|---|---|---|
| **B1 (Baseline)** | *"I'm really sorry you're going through this. You're not alone in this."* | 0.611 | 0.092 |
| **ECOT-T5** | *"I'm so sorry. Losing a pet is a real grief, and there's no timeline for it. It makes complete sense that you're hurting right now."* | 0.745 | 0.312 |

Strategy applied: **Validation** (uncontrollable)

### Example 3 — Controllable / Positive Valence

> **Utterance**: "I finally got my promotion after working for it for two years."

| System | Response | BERTScore | EASE |
|---|---|---|---|
| **B1 (Baseline)** | *"I'm really sorry you're going through this. You're not alone in this."* | 0.589 | 0.054 |
| **ECOT-T5** | *"That's a huge achievement, especially after putting in that kind of sustained effort. You should feel really proud of yourself."* | 0.711 | 0.288 |

Strategy applied: **Reframing** (controllable, positive)

---

## Failure Analysis

Three failure mode categories identified in ECOT-T5:

| Failure Mode | Description | Frequency |
|---|---|---|
| **Step-2 hallucination** | Dominant-cause paraphrase modifies meaning of step-1 cause (e.g., "Such good memories" → "Such bad memories") | ~15% of cases |
| **Strategy–controllability mismatch** | Selected strategy does not align with predicted controllability label | ~8% of uncontrollable instances |
| **Generic response collapse** | Despite correct cause and strategy, response defaults to high-frequency generic form | ~3% of all outputs |

---

## Implementation Details

### Hyperparameters

| Parameter | Track A (T5-base) | Track C (LLaMA-2-7B) | Controllability Clf |
|---|---|---|---|
| Base Checkpoint | `google/flan-t5-base` | `meta-llama/Llama-2-7b-chat-hf` | `roberta-base` |
| Learning Rate | 2×10⁻⁵ | 5×10⁻⁵ | 2×10⁻⁵ |
| LR Scheduler | Cosine | Cosine | Linear |
| Epochs | 1 | 1 | 10 |
| Effective Batch Size | 16 | 16 | 16 |
| Optimizer | AdamW | Paged AdamW (8-bit) | AdamW |
| Max Input Length | 512 tokens | 160 tokens | 128 tokens |
| DPO β | 0.5 | 0.1 | — |
| LoRA Rank (r) | — | 16 | — |
| LoRA α | — | 32 | — |
| λ₁ (Ctrl Loss Weight) | 0.3 | — | — |
| λ₂ (DPO Loss Weight) | 0.5 | — | — |
| Random Seed | 42 | 42 | 42 |

**Hardware**: Nvidia K80/T4 GPU (80 GB RAM), PyTorch. Fine-tuning averaged ~15 minutes per model.

---

## Ethical Considerations

- **Strategy-controllability mismatch risk**: Could be psychologically harmful if deployed in sensitive mental health contexts without additional safeguards.
- **Binary controllability labels**: May reflect localized cultural/demographic assumptions of annotators; "controllable error" may not generalize equitably across globally diverse societies.
- **AI Use**: AI assistants (Claude) were used for grammar checking and sentence-level editing during manuscript preparation. All scientific content, experimental design, results, and conclusions are entirely the authors' own.
- **Dataset usage**: All datasets and models are used solely for research purposes. Derived datasets are intended for research use only and should not be deployed in clinical or commercial settings without further validation.

---

## License & Artifacts

| Artifact | License |
|---|---|
| EmpatheticDialogues | CC BY 4.0 |
| T5 | Apache License 2.0 |
| LLaMA-2 | Meta Custom License |
| Sentence Transformers | Apache License 2.0 |
| MIME / MoEL | MIT License |
| ECOT Preference Dataset | Available at [anonymous repository](https://anonymous.4open.science/r/ECOT-ERG-DBCF/) |
| EASE Metric | Available at [anonymous repository](https://anonymous.4open.science/r/easekit-0DCE/README.md) |

---

## Citation

> This paper is currently under blind review. Citation will be updated after publication.

```bibtex
@article{ecot-erg-2026,
  title   = {Towards Causally Faithful Empathetic Dialogue Generation via Structured Appraisal Reasoning},
  author  = {Gupta, Srishti and Sarkar, Ayushman and Dutta, Sayan and Vishwakarma, Shashank and Raj, Rishikesh and Jana, Arghya and Teja, Nagandla Tarun},
  journal = {ACL ARR 2026 Submission},
  year    = {2026},
  note    = {Under Review. Preferred Venue: EMNLP}
}
```

---

## Authors

Srishti Gupta · Ayushman Sarkar · Sayan Dutta · Shashank Vishwakarma · Rishikesh Raj · Arghya Jana · Nagandla Tarun Teja

**Country of Origin**: UAE · **Submission #12976** · **Submitted**: 26 May 2026
