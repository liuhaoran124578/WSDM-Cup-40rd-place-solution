#!/bin/bash
set -e


qwen_path=../model_path/qwen2.5_72b
llama_path=../model_path/llama3.1_70b
gemma_path=../model_path/Gemma2_9b_it_simpo
deepseekr1_path= ../model_path/deepseekr1_70b



python prepare_data.py ${qwen_path} qwen2.5
python prepare_data.py ${llama_path} llama3.1
python prepare_data.py ${gemma_path} gemma2
python prepare_data.py ${deepseekr1_path} deepseekr1




python prepare_data_ut.py ${qwen_path} qwen2.5
python prepare_data_ut.py ${llama_path} llama3.1
python prepare_data_ut.py ${gemma_path} gemma2
python prepare_data_ut.py ${deepseekr1_path} deepseekr1

