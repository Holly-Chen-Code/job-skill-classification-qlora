# QLoRA Fine-Tuning for Functional Job Skill Classification

---
## Research and Selection of Methods

### Define Objectives

The objective of this project is to investigate whether a small language model can accurately classify the functional skill category of a job posting.

Given a job title and job description, the model is trained to predict the corresponding `skill_name`. This task is formulated as supervised instruction fine-tuning for text classification using a causal language model.

---
### Literature Review

Recent studies have shown that parameter-efficient fine-tuning methods such as LoRA and QLoRA can significantly reduce GPU memory requirements while maintaining competitive performance.

The Phi-3 Mini model was selected because it provides strong language understanding capabilities with a relatively small number of parameters, making it suitable for educational projects and limited GPU resources.

The project is implemented using Hugging Face Transformers and PEFT, following current best practices for efficient instruction fine-tuning.

---
### Benchmarking

Several approaches were considered during the project design.

Traditional machine learning methods such as TF-IDF with Logistic Regression are computationally efficient but have limited contextual understanding.

Large language models provide stronger semantic reasoning but often require substantial computational resources.

Phi-3 Mini combined with QLoRA offers a practical balance between model performance, memory efficiency, and training cost, making it suitable for this project.

---
### Preliminary Experiments

Before training on the complete dataset, a development subset was used to validate the entire pipeline.

The preliminary experiment confirmed that:

- Data preprocessing was successful.
- Instruction-response formatting worked correctly.
- The QLoRA fine-tuning pipeline completed without errors.
- Training loss decreased over one epoch.
- A sample inference correctly predicted the expected skill category.

These experiments verified the feasibility of the proposed approach before larger-scale training.

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
