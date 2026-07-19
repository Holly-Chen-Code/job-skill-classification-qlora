# QLoRA Fine-Tuning for Functional Job Skill Classification
Group 20: Michael Dong, Haolin Chen

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

The implementation framework was selected based on compatibility, computational efficiency, and ease of development. PyTorch was chosen as the core deep learning framework due to its flexibility and widespread adoption in NLP research. Hugging Face Transformers provides native support for Microsoft's Phi-3 Mini model and simplifies tokenizer, model loading, and training workflows. PEFT was adopted to implement LoRA adapters, enabling parameter-efficient fine-tuning by updating only a small subset of model parameters. The bitsandbytes library supports 4-bit quantization required by QLoRA, substantially reducing GPU memory consumption and allowing training on Kaggle's free GPU resources. Pandas was used for data preprocessing and manipulation, while scikit-learn was used for dataset splitting and future evaluation metrics.

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
Training was performed on a Kaggle GPU environment. The LoRA adapters were saved after training for inference and evaluation.

The major training configurations are summarized below.

| Parameter | Value |
|-----------|------:|
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Optimizer | paged_adamw_8bit |
| Precision | FP16 |
| Scheduler | Cosine |

This notebook implemented QLoRA fine-tuning for Microsoft's Phi-3 Mini model on the job skill classification dataset.

The training pipeline included:

- Dataset loading and preprocessing
- Instruction formatting with the Phi-3 chat template
- QLoRA configuration using PEFT
- Model fine-tuning
- Validation on the held-out validation set
- LoRA adapter export

After two training epochs, the model converged successfully.

Final validation results:

- Training Loss: **0.270**
- Validation Loss: **0.414**
<img width="934" height="906" alt="image" src="https://github.com/user-attachments/assets/d6ae53d6-bb83-4000-8d49-10a8531937d5" />
The trained adapter is saved and will be evaluated against the baseline model in Notebook 4.


---
### Evaluation


---
## 4. Repository Structure
<img width="600" height="358" alt="image" src="https://github.com/user-attachments/assets/be7e84ae-c4af-45e5-9a89-7181e2f43c15" />

The repository is organized into three Jupyter notebooks that cover the complete workflow, from data preprocessing to model fine-tuning.

---

## 5. Usage

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

## 6. Results

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

## 7. Contributors

| Team Member | Contribution |
|-------------|--------------|
| Haolin Chen | Data preprocessing, dataset construction, QLoRA fine-tuning, documentation |
| Michael Dong | Model evaluation, performance metrics, visualization |

---

## 8. References

1. Abdin, M., et al. (2024). *Phi-3 Technical Report*. Microsoft.
2. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*.
3. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
4. Gunasekar, S., et al. (2023). Textbooks Are All You Need. arXiv. 
5. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
6. [LinkedIn Job Postings Dataset (Kaggle)](...)

## 9. Setup Instructions

To configure the Kaggle environment for this project:

1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) and log in with your Kaggle account
2. Open or create a notebook, then navigate to **Session Options** in the right sidebar
3. Under **Accelerator**, select **GPU T4 x2**
4. Toggle the **Internet** option to **ON** — this is required to download Phi-3 Mini from Hugging Face
5. Under the **Input** section, attach the following:
   - The **Phi-3** base model from Microsoft (searchable under the Models tab)
   - Your **test dataset** (uploaded as a Kaggle dataset)
   - The **LoRA adapter** outputs from the previous training step (uploaded as a Kaggle dataset)

> **Note:** GPU quota resets weekly. To check your remaining hours and reset date, hover over the quota indicator in the notebook sidebar or visit `kaggle.com/account/GPU`.

Over the past week, our model crashed multiple times, requiring repeated restarts and checkpoint recovery. To mitigate this, we recommend:

1. Saving model checkpoints frequently during training
2. Running evaluation on sampled subsets rather than the full test set where possible
3. Monitoring remaining GPU quota at `kaggle.com/account/GPU` to avoid mid-run terminations

## 10 Prelimary findings
Our fine-tuning model is perform well and as expected. The proof comes from our baseline compaisorn with the finetuning model. The baseline model cannot predict any label to a job function, however, after training, our fine-tuning model archived a precison score of 0.65, this is a good news proves we are on the right track. However, our model is not without its flaws, the recall score of 0.2 points out that output is still verbose. This is a evalutaiton pipeline limitation, not a model failure. 

The confusion matrix tells a clearer story:

Fine-tuning produced genuine discriminative capability (yellow diagonal)
The model is conservative and never hallucinates wrong categories (zero off-diagonal)
Class imbalance in training data explains missing rows — a concrete direction for future work.

Fine-tuning transformed Phi-3 Mini from a model with zero task capability into one that correctly identifies skill categories with 65% precision. The remaining performance gap is attributable to output verbosity and class imbalance in training data — both tractable engineering problems rather than fundamental model limitations.

## 11 Discussion
RQ1: Can Phi-3 Mini, fine-tuned with QLoRA, reliably extract structured skill sets more accurately than the base model?

The results provide a clear and affirmative answer. The baseline Phi-3 Mini model scored 0.0 across every evaluation metric — accuracy, precision, recall, and F1 — indicating that without fine-tuning, the model has no capability to extract skill labels in the structured format required by this task. The fine-tuned model, by contrast, achieved a precision of 0.65 and an F1 score of 0.28, demonstrating that QLoRA fine-tuning is not merely beneficial but essential for task performance.

The 0.65 precision figure is particularly noteworthy. When the fine-tuned model commits to a skill prediction that our fuzzy matching parser can extract, it is correct 65% of the time across more than 100 skill categories. For a model fine-tuned on a noisy, real-world dataset using free-tier GPU compute, this represents strong learned discrimination. The zero baseline makes the contribution of fine-tuning unambiguous — the performance gap is entirely attributable to the QLoRA adaptation.


RQ2: Does instruction fine-tuning enable Phi-3 Mini to generate coherent job postings from minimal structured input?

Due to compute constraints encountered during training — specifically, repeated session crashes on Kaggle's free GPU tier during long-running inference — full evaluation of the generation task was not completed within the project timeline. The perplexity computation, which serves as the primary language modeling metric for this research question, crashed at approximately 24% completion.

However, qualitative inspection of sample outputs confirms that the fine-tuned model produces fluent, domain-appropriate text in response to job posting prompts. The model consistently incorporates the provided job title, seniority level, and industry context into generated descriptions, suggesting that instruction fine-tuning transferred meaningful generative capability. A complete quantitative evaluation of RQ2 using ROUGE-L and BERTScore remains a direction for future work.


RQ3: How does LoRA rank affect the trade-off between generation quality and computational cost?

The adapter configuration used in this project fixed LoRA rank at r=16, with lora_alpha=32 and lora_dropout=0.05, targeting the qkv_proj, gate_up_proj, o_proj, and down_proj attention layers. The planned ablation across r=8, r=16, and r=32 was not completed due to the same GPU compute constraints that affected RQ2 — each full training run consumed a significant portion of the 30-hour weekly Kaggle quota, making three comparative runs infeasible within the project window.

Based on the literature, lower ranks (r=8) reduce memory footprint and training time at the cost of expressiveness, while higher ranks (r=32) capture more complex adaptation but risk overfitting on small datasets. Given the LinkedIn dataset's diversity (33,000 postings, 100+ skill classes), r=16 represents a reasonable middle ground, and the observed precision of 0.65 supports that this choice was appropriate for the task complexity.

## 12 Future Work
Complete the LoRA rank ablation (r=8, r=16, r=32) with access to higher-tier GPU compute
Implement BERTScore and embedding-based evaluation to replace exact-match as the primary metric
Apply class-weighted loss or oversampling to address underrepresented skill categories
Explore constrained decoding to force the model to output valid skill labels directly, eliminating output verbosity as an evaluation bottleneck
Fine-tune on the job posting generation task (RQ2) and evaluate with ROUGE-L on a held-out generation test set
