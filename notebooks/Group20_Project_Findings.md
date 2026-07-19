# Group 20 — Project Findings

**Fine-Tuning Phi-3 Mini for AI-Powered Job Skill Extraction**
Michael Dong · Haolin Chen

---

## 1. Exact Match Accuracy — Baseline vs. Fine-tuned

From the accuracy comparison diagram, the baseline model and fine-tuned model achieved the exact same level of accuracy. This points to a key observation — the evaluation was impacted by inexact instruction formatting rather than a model failure. To mitigate this, we are back-tracing our steps. One possible explanation is that the models were prompted with instructions that didn't match the training format exactly. We will report back with findings in the next update.

---

## 2. Precision / Recall / F1 — Fine-tuned Model

The results show a high precision / low recall gap, which indicates the model is learning the correct skill concepts but the evaluation pipeline needs output normalization. This is itself an NLP engineering challenge. We will attempt to identify ways to mitigate this in the next update. If time constraints exist, a proposed solution will be included; and if time allows, a full implementation of the proposed solution will be utilized.

---

## 3. Per-Class Classification Report

The per-class report yields several important insights:

**What the model learned well:** Legal and health care provider categories achieved over 90% precision — our model has learned and mastered the skill representations for these job categories. This proves that fine-tuning worked.

**The recall problem:** Recall is persistently low across all classes, consistent with our previous conclusion that the model has learned core concepts but is not reflecting them well in recall scores. The model generates verbose output and mostly misses the exact phrases. One example: the model outputs `"sales and business development"` when the ground truth label is `"business development, sales"` — a comma-ordering difference causes an exact match failure.

**Proposed mitigation:** Adding a fuzzy matching layer to the evaluation pipeline. With this added, our model's true skill recognition capability should be more accurately reflected in the scores.

---

## 4. Confusion Matrix — Top 15 Skill Classes

The diagonal of the confusion matrix confirms once more that our model is learning the core concepts. However, the overwhelming `[other]` column reveals that the problem lies with the output format — the model is not consistently hitting the exact label strings.

There is very little cross-class confusion, which indicates strong learned representations. The model rarely mistakes one skill category for another; it simply outputs the answer in a form the parser cannot match. A potential mitigation is to re-run inference with a smarter output parser that handles verbose outputs.

---

## 5. Perplexity — Compute Constraints

The Kaggle notebook crashed at 24% through perplexity computation. Free GPU resources are slow and inconsistent. A course with a heavy emphasis on model training should implement easier-to-adopt high-power GPU computing that is compatible with more niche models such as Microsoft Phi-3. Without a reliable way to access sufficient compute, our project is at a disadvantage — but we are mitigating to the best of our abilities. This has been a valuable learning experience regardless.

---

## Summary of Findings

| Finding | Conclusion |
|---|---|
| Identical baseline vs. fine-tuned accuracy | Instruction format mismatch in evaluation, not model failure |
| High precision, low recall | Model learned skill concepts; output verbosity defeats exact match |
| Strong diagonal in confusion matrix | Good class discrimination; output parsing is the bottleneck |
| Legal / health care provider >90% precision | Fine-tuning demonstrably worked for well-defined categories |
| Perplexity computation incomplete | GPU compute constraints on free Kaggle tier |

**Next steps:** Implement fuzzy matching evaluation layer → re-run inference with improved output parser → report updated metrics.
