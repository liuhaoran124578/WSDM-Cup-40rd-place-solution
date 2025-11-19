#!/bin/bash
set -e

qwen_path=../model_path/qwen2.5_72b
llama_path=../model_path/llama3.1_70b
gemma_path=../model_path/Gemma2_9b_it_simpo
# deepseek_path=../model_path/deepseekr1_70b

qwen_path_ut=../model_save/qwen2.5_4bit_pretrain/epoch_0_model/adapter.bin
llama_path_ut=../model_save/llama3.1_4bit_pretrain/epoch_0_model/adapter.bin
gemma_path_ut=../model_save/gemma2_4bit_pretrain/epoch_0_model/adapter.bin
# deepseek_path_ut=../model_save/deepseekr1_4bit_pretrain/epoch_0_model/adapter.bin


fold=$1
echo run:${fold}
# train llama3.1 70b
sh run_fintune.sh llama3.1 ${llama_path}  ${llama_path_ut} ${fold}
# predict train logits
python predict_train.py ${llama_path} ../model_save/llama3.1_4bit_load_fintune/epoch_0_model/adapter.bin ../data/processed_data/llama3.1fold${fold}/train.pkl ../data/oof/llama3.1fold${fold}_train.pkl

# train qwen2.5 72b
sh run_fintune.sh qwen2.5 ${qwen_path}  ${qwen_path_ut} ${fold}
# predict train logits
python predict_train.py ${qwen_path} ../model_save/qwen2.5_4bit_load_fintune/epoch_0_model/adapter.bin ../data/processed_data/qwen2.5fold${fold}/train.pkl ../data/oof/qwen2.5fold${fold}_train.pkl

# merge  logits 
python merge_logits.py ../data/processed_data/gemma2fold${fold}/train.pkl ../data/oof/qwen2.5fold${fold}_train.pkl ../data/oof/llama3.1fold${fold}_train.pkl ../data/processed_data/gemma2fold${fold}/train_logits.pkl

# distill fintune gemma2-9b-it-simpo
sh run_fintune_16bit_distill.sh gemma2 ${gemma_path} ${gemma_path_ut} ${fold}
