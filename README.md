# QLoRA Fine-Tuning for Functional Job Skill Classification

---
## 1.Project overview
This project fine-tunes Microsoft's Phi-3 Mini model using QLoRA to classify the functional skill category of a job posting.

After exploring the LinkedIn Job Postings dataset, we refined the scope of our original project proposal to better align with the available data. Instead of generating job descriptions, our project focuses on supervised instruction fine-tuning for text classification. The model learns to predict the correct `skill_name` using the job title and job description.

The project demonstrates how a small language model can perform structured skill classification using real-world LinkedIn job postings.

---

## 2.Research and Selection of Methods

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
## 3.Implementation

---
### Framework Selection
The project was implemented using the following frameworks and libraries:

| Framework / Library | Purpose |
|---------------------|---------|
| PyTorch | Deep learning framework used for model training |
| Hugging Face Transformers | Loading and fine-tuning the pretrained Phi-3 Mini model |
| PEFT | Parameter-efficient fine-tuning using LoRA |
| bitsandbytes | 4-bit quantization for QLoRA training |
| pandas | Data preprocessing and dataset manipulation |
| scikit-learn | Train/validation/test splitting and evaluation utilities |

The Hugging Face ecosystem was selected because it provides seamless support for pretrained language models and integrates well with PEFT for parameter-efficient fine-tuning. PyTorch offers flexibility during model training, while bitsandbytes enables efficient 4-bit quantization, making it possible to fine-tune Phi-3 Mini on a Kaggle GPU.

---
### Dataset Preparation
<img width="510" height="746" alt="image" src="https://github.com/user-attachments/assets/fd8f908d-1a2e-4552-94f1-f6b01b8d42f9" />
             
The project uses the LinkedIn Job Postings dataset obtained from Kaggle. Three related tables were used:

- `postings.csv` – job posting information, including job titles and descriptions
- `job_skills.csv` – mappings between job postings and skill identifiers
- `skills.csv` – skill identifiers and their corresponding skill names

The `postings.csv` table was first explored and cleaned. Records with missing or unusable job titles and descriptions were removed, duplicate records were addressed, and the relevant columns were selected.

After cleaning, the processed postings data was joined with `job_skills.csv` and `skills.csv` to associate each job posting with its corresponding `skill_name`.

The merged dataset was then transformed into an instruction-following format. Each example contains the job title and job description as the model input and the corresponding `skill_name` as the target output. Finally, the dataset was divided into training, validation, and test sets for model development and evaluation.

---
### Model Development

The project fine-tunes Microsoft's Phi-3 Mini, a lightweight decoder-only transformer designed for instruction-following tasks.

Instead of updating all model parameters, the project adopts QLoRA, which injects Low-Rank Adaptation (LoRA) layers into the pretrained model. This significantly reduces the number of trainable parameters while maintaining competitive performance.

The model receives the job title and job description as input and generates the corresponding `skill_name` as the output.
<img width="216" height="364" alt="image" src="https://github.com/user-attachments/assets/833465c1-71ac-445f-9ab0-85aaad070e94" />


 ---
### Training & Fine-tuning

The model was fine-tuned using QLoRA with the Hugging Face Trainer API.

The major training configurations are summarized below.

| Parameter | Value |
|-----------|------:|
| Epochs | 1 |
| Learning Rate | 2e-4 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Optimizer | paged_adamw_8bit |
| Precision | FP16 |
| Scheduler | Cosine |

Training was performed on a Kaggle GPU environment. The LoRA adapters were saved after training for inference and evaluation.
<img width="1152" height="494" alt="image" src="https://github.com/user-attachments/assets/36525085-cf52-43e4-b880-4e85500e9b89" />
<img width="735" height="455" alt="image" src="https://github.com/user-attachments/assets/cb5bc1b3-7ab5-4f44-b4fc-cb357f1f435d" />

---
### Evaluation


---
## Repository Structure
<img width="600" height="358" alt="image" src="https://github.com/user-attachments/assets/be7e84ae-c4af-45e5-9a89-7181e2f43c15" />

The repository is organized into three Jupyter notebooks that cover the complete workflow, from data preprocessing to model fine-tuning.

```
## Usage

Run the notebooks in the following order:

1. `01_data_preprocessing.ipynb`
   - Explore and clean the LinkedIn Job Postings dataset.
   - Merge the three tables.

2. `02_build_training_dataset.ipynb`
   - Convert the merged dataset into instruction-response pairs.
   - Split the dataset into training, validation, and test sets.

3. `03_phi3_qlora_training.ipynb`
   - Fine-tune Microsoft's Phi-3 Mini using QLoRA.

4. `04_model_evaluation.ipynb`
   - Evaluate the fine-tuned model and calculate performance metrics.

---
## Results

The preliminary implementation successfully completed the end-to-end training pipeline.

Current progress includes:

- Successfully merged and cleaned the LinkedIn Job Postings dataset.
- Constructed instruction-response training examples.
- Fine-tuned Phi-3 Mini using QLoRA.
- Verified that the model can generate the expected `skill_name` for sample job postings.

The following figures illustrate the current implementation results.

- Training Loss
<img width="928" height="377" alt="image" src="https://github.com/user-attachments/assets/a5ad561e-3433-4602-a88f-2520829a9ad4" />
- Validation Loss
<img width="699" height="455" alt="image" src="https://github.com/user-attachments/assets/719eae6e-123e-469a-9da3-1785528aa6b0" />
- Sample Prediction
<img width="970" height="269" alt="image" src="https://github.com/user-attachments/assets/c5376e9d-3464-4830-b44d-8c28717c4985" />

---
## Contributors

| Team Member | Contribution |
|-------------|--------------|
| Haolin Chen | Data preprocessing, dataset construction, QLoRA fine-tuning, documentation |
| Michael Dong | Model evaluation, performance metrics, visualization |

---
## References

1. Abdin, M., et al. (2024). *Phi-3 Technical Report*. Microsoft.
2. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.
3. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
4.[Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
5.[LinkedIn Job Postings Dataset (Kaggle)](...)

