# QLoRA Fine-Tuning for Functional Job Skill Classification

## Project Overview

This project fine-tunes a large language model using QLoRA to identify the functional skill category of a job posting.

The model takes a job title and job description as input and predicts the corresponding functional skill category.

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
| [03_qlora_fine_tuning.ipynb](notebooks/03_qlora_fine_tuning.ipynb) |
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
