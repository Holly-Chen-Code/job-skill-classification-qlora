# QLoRA Fine-Tuning for Functional Job Skill Classification

---
## Research and Selection of Methods

### Define Objectives

This project operates in the Natural Language Processing (NLP) domain, with a focus on supervised instruction fine-tuning of a small language model (SLM) for job skill classification. The objective is to train the model to predict the appropriate functional skill category (`skill_name`) based on a job title and job description.

After exploring the LinkedIn Job Postings dataset, we refined the scope of our original proposal to better align with the available data. Instead of conditional text generation, the project focuses on text classification using instruction tuning.

---
### Literature Review

Recent research on large language models has primarily focused on scaling model size to improve performance. Models such as GPT-3 and GPT-4 demonstrate strong language understanding capabilities but require substantial computational resources for both training and deployment.

The Microsoft Phi series explores a different direction by emphasizing smaller yet highly capable language models. Similar to GPT-style models, the Phi family adopts a Transformer-based autoregressive architecture while relying on high-quality training data and efficient scaling strategies. Previous studies have shown that textbook-quality datasets and reasoning-focused optimization can significantly improve the performance of small language models (Gunasekar et al., 2023; Li et al., 2023).

Phi-3 further improves instruction-following capability and supports efficient deployment on consumer-grade hardware, making it well suited for parameter-efficient fine-tuning approaches such as LoRA and QLoRA (Abdin et al., 2024).

---
### Benchmarking

Several approaches were considered during the project design.

Traditional machine learning methods such as TF-IDF combined with Logistic Regression provide efficient text classification but have limited semantic understanding.

Large language models offer stronger contextual reasoning but often require significant computational resources.

Considering model capability, computational efficiency, and GPU limitations, Microsoft Phi-3 Mini was selected as the base model. Combined with QLoRA, the model can be fine-tuned efficiently while updating only a small subset of trainable parameters.

---
### Preliminary Experiments


Before training on the complete dataset, a development subset was used to validate the entire pipeline.

The preliminary experiment confirmed that:

- The LinkedIn job posting tables were successfully joined and cleaned.
- Instruction-response training pairs were generated correctly.
- The dataset was split into training, validation, and test sets.
- The QLoRA fine-tuning pipeline completed successfully.
- Training loss decreased over one epoch while validation loss remained stable.
- A sample inference correctly predicted the expected `skill_name`.

These experiments verified the feasibility of the proposed approach before larger-scale experiments.
---

## Dataset

This project uses the **LinkedIn Job Postings** dataset from Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

---

## Repository Structure

### Notebooks

| Notebook | Description |
|-----------|-------------|
| [01_data_preprocessing.ipynb](notebooks/01_data_preprocessing.ipynb) | Data cleaning and preprocessing |
| [02_build_training_dataset.ipynb](notebooks/02_build_training_dataset.ipynb) | Build instruction-response training dataset |
| [03_qlora_fine_tuning.ipynb](notebooks/03_qlora_fine_tuning.ipynb) | Fine-tune the model using QLoRA |
| [04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb) | Evaluate the fine-tuned model |

### Other Directories

| Folder | Description |
|--------|-------------|
| [data](data) | Dataset description |
| docs | Project proposal and documentation |
| outputs | Model outputs and evaluation results |

---

## Project Workflow

```text
LinkedIn Job Postings
        ↓
Data Cleaning
        ↓
Instruction Dataset
        ↓
Phi-3 + QLoRA Fine-tuning
        ↓
Skill Prediction
        ↓
Model Evaluation
```
