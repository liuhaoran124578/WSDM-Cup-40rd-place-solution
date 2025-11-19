# WSDM-Cup-40rd-place-solution

This repository contains the source code for the 40th place solution in the WSDM Cup. The solution involves a multi-stage training pipeline including post-pretraining, knowledge distillation, and fine-tuning using LoRA.

## 🖥 Hardware Specifications

The solution was trained on the following hardware setup:

- **CPU Cores**: 128
- **Memory**: 768 GB
- **GPU**: NVIDIA Tesla A100 80G
- **Number of GPUs**: 8
- **OS/Platform**: Linux

## 🛠 Environment & Requirements

### Third-Party Software
- **Python**: 3.10.14
- **PyTorch**: 2.3.1+cu121
- **CUDA**: 12.2
- **cuDNN**: 8.9.2.26

### Installation
To set up the environment, please install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Directory Structure

```text
├── README.md
├── code
│   ├── data                # Train data and other datasets
│   ├── model_save_or       # Path for saving original/intermediate models
│   ├── src                 # Main source code for the solution
│   │   ├── merge_logits.py
│   │   ├── merge_lora.py
│   │   ├── notebook        # Jupyter notebooks for preprocessing & analysis
│   │   │   ├── preprocess_data.ipynb
│   │   │   ├── process_finetune_data.ipynb
│   │   │   ├── process_refinetune.ipynb
│   │   │   └── withdraw_train_71k.ipynb
│   │   ├── predict_train.py
│   │   ├── prepare_data.py
│   │   ├── prepare_data_ut.py
│   │   ├── scripts         # Shell scripts to run the pipeline stages
│   │   │   ├── run_data.sh
│   │   │   ├── run_fintune.sh
│   │   │   ├── run_fintune_16bit_distill.sh
│   │   │   ├── run_pipeline.sh
│   │   │   ├── run_post_pretrain.sh
│   │   │   └── run_post_pretrain_ut_four_model.sh
│   │   ├── train_ce_model_nice_memory.py
│   │   └── train_ce_model_nice_memory_grad_gemma2_double_16bit.py
│   └── sub                 # Submission generation
├── requirements.txt
└── review
```

## 📥 Model Download Preparation

Please download the following four pretrained models and place them in the `model_path` directory (create this folder inside `code/` or configure your path accordingly):

1.  **Llama3.1 70b**
    *   Link: [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
    *   **Rename folder to**: `llama3.1_70b`

2.  **Qwen2.5 72b**
    *   Link: [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
    *   **Rename folder to**: `qwen2.5_72b`

3.  **Gemma2-9b-it-simpo**
    *   Link: [princeton-nlp/gemma-2-9b-it-SimPO](https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO)
    *   **Rename folder to**: `Gemma2_9b`

4.  **DeepSeek-R1 70b**
    *   Link: [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
    *   **Rename folder to**: `deepseek_r1`

## 🚀 Training Pipeline

*Note: The time estimates mentioned below are based on a single machine with 8 NVIDIA A100 80GB GPUs.*

### 1. Preprocess Data
Convert the raw data into a format suitable for training.

```bash
cd code/src
sh scripts/run_data.sh
```

### 2. Post-Pretrain
Use UT (Unlabeled Text) data to perform post-pretraining on the four models (Llama, Qwen, Gemma, DeepSeek).

```bash
# Ensure you are in code/src
sh scripts/run_post_pretrain_ut_four_model.sh
```

### 3. Fine-Tune Models (5-Fold Strategy)
We use a 5-fold cross-validation strategy. Each fold includes:
1.  Training Llama3.1 and Qwen2.5 (DeepSeek R1) models.
2.  Predicting to obtain the probability distribution for the training set.
3.  Fine-tuning the Gemma model using Knowledge Distillation.

Run the pipeline for each fold:

```bash
# Run sequentially or in parallel if resources allow
sh scripts/run_pipeline.sh 0
sh scripts/run_pipeline.sh 1
sh scripts/run_pipeline.sh 2
sh scripts/run_pipeline.sh 3
sh scripts/run_pipeline.sh 4
```

#### Knowledge Distillation Details
During the fine-tuning phase, we utilize a combined loss function including Cross Entropy, KL Divergence (for distillation), and Cosine Embedding Loss.

```python
loss_fun = nn.CrossEntropyLoss()
divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
cos_loss_fn = nn.CosineEmbeddingLoss()

outputs = model(batch['input_ids'], use_cache=False) # predict gemma2
logits = outputs.logits
grads = batch['grads']
grads1 = batch['grads'][:, :2] # qwen2.5 
grads2 = batch['grads'][:, 2:] # llama3
labels = batch['labels']

# 1. Cross Entropy Loss
loss_ce = loss_fun(logits, labels)

# 2. Distillation from Qwen2
loss_grad1 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads1 / T, dim=1)
)
cos_loss1 = cos_loss_fn(F.softmax(grads1 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

# 3. Distillation from Llama3
loss_grad2 = divergence_loss_fn(
    F.log_softmax(logits / T, dim=1),
    F.softmax(grads2 / T, dim=1)
)
cos_loss2 = cos_loss_fn(F.softmax(grads2 / T, dim=1), F.softmax(logits / T, dim=1),
                        torch.ones(logits.size()[0]).to(logits.device))

# Total Loss
loss = (loss_ce + loss_grad1 + cos_loss1 + loss_grad2 + cos_loss2) / 5.
```

### 4. Merge LoRA and Quantize
Finally, merge the LoRA layers of the 5-fold Gemma models and quantize them to 8-bit for inference/submission.