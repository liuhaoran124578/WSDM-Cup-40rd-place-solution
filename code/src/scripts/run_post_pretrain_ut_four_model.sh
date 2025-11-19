#!/bin/bash
set -e


qwen_path=../model_path/qwen2.5_72b
llama_path=../model_path/llama3.1_70b
gemma_path=../model_path/Gemma2_9b_it_simpo
deepseekr1_path=../model_path/deepseekr1_70b

sh run_post_pretrain.sh llama3.1 ${llama_path}
sh run_post_pretrain.sh qwen2.5 ${qwen_path}
sh run_post_pretrain.sh gemma2 ${gemma_path}
sh run_post_pretrain.sh deepseekr1 ${deepseekr1_path}





