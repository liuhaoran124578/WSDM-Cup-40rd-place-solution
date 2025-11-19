#!/usr/bin/env python

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pickle
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import sys

model_path = sys.argv[1]
save_name = sys.argv[2]
print("model_path:", model_path)
print("save_name:", save_name)

### load tokenizer
MODEL_NAME = model_path
MAX_LENGTH = 3072
MAX_PROMPT_LENGTH = 768
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_eos_token = True
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# load data
train = pd.read_csv("../data/wsdm_competition/ut_data.csv")

def do(row):
    if row['winner']=='model_a':
        return "A"
    else:
        return "B"

train['label'] = train.apply(lambda row: do(row), axis=1)
print(train['label'].value_counts())

### load template
################################# gemma2 template ########################################################
if save_name == 'gemma2':
    template_1 = (
        "<start_of_turn>user\n"
        "Act as an impartial judge and evaluate the quality of responses A and B to the user question. "
        "Choose the response that better follows the user’s instructions, considering factors such as helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail. Be aware that the prompt and responses may be "
        "incomplete due to input length limitations. The evaluation should not be influenced by position biases, "
        "presentation order, or response lengths. Do not favor specific assistant names.\n"
    )
    template_2 = "Question:"
    template_3 = "\n"
    template_4 = "Response A:"
    template_5 = "\n"
    template_6 = "Response B:"
    template_7 = "<end_of_turn>\n<start_of_turn>model"

    template_1tokenized = tokenizer(template_1, add_special_tokens=False)["input_ids"]
    template_2tokenized = tokenizer(template_2, add_special_tokens=False)["input_ids"]
    template_3tokenized = tokenizer(template_3, add_special_tokens=False)["input_ids"]
    template_4tokenized = tokenizer(template_4, add_special_tokens=False)["input_ids"]
    template_5tokenized = tokenizer(template_5, add_special_tokens=False)["input_ids"]
    template_6tokenized = tokenizer(template_6, add_special_tokens=False)["input_ids"]
    template_7tokenized = tokenizer(template_7, add_special_tokens=False)["input_ids"]

################################# qwen2.5 template ########################################################
elif save_name == 'qwen2.5':
    template_1 = (
        "<im_start>system\n"
        "Act as an impartial judge and evaluate the quality of responses A and B to the user question. "
        "Choose the response that better follows the user’s instructions, considering factors such as helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail. Be aware that the prompt and responses may be "
        "incomplete due to input length limitations. The evaluation should not be influenced by position biases, "
        "presentation order, or response lengths. Do not favor specific assistant names.<im_end>\n"
    )
    template_2 = "<im_start>user\nQuestion:"
    template_3 = "\n"
    template_4 = "Response A:"
    template_5 = "\n"
    template_6 = "Response B:"
    template_7 = "<im_end>\n<|im_start|>assistant"

    template_1tokenized = tokenizer(template_1, add_special_tokens=False)["input_ids"]
    template_2tokenized = tokenizer(template_2, add_special_tokens=False)["input_ids"]
    template_3tokenized = tokenizer(template_3, add_special_tokens=False)["input_ids"]
    template_4tokenized = tokenizer(template_4, add_special_tokens=False)["input_ids"]
    template_5tokenized = tokenizer(template_5, add_special_tokens=False)["input_ids"]
    template_6tokenized = tokenizer(template_6, add_special_tokens=False)["input_ids"]
    template_7tokenized = tokenizer(template_7, add_special_tokens=False)["input_ids"]

################################# deepseekr1 template ########################################################
elif save_name == 'deepseekr1':
    template_1 = (
        "<｜begin▁of▁sentence｜>"
        "Act as an impartial judge and evaluate the quality of responses A and B to the user question. "
        "Choose the response that better follows the user’s instructions, considering factors such as helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail. Be aware that the prompt and responses may be "
        "incomplete due to input length limitations. The evaluation should not be influenced by position biases, "
        "presentation order, or response lengths. Do not favor specific assistant names."
    )
    template_2 = "<｜User｜>Question:"
    template_3 = "\n"
    template_4 = "Response A:"
    template_5 = "\n"
    template_6 = "Response B:"
    template_7 = "<｜Assistant｜>"

    template_1tokenized = tokenizer(template_1, add_special_tokens=False)["input_ids"]
    template_2tokenized = tokenizer(template_2, add_special_tokens=False)["input_ids"]
    template_3tokenized = tokenizer(template_3, add_special_tokens=False)["input_ids"]
    template_4tokenized = tokenizer(template_4, add_special_tokens=False)["input_ids"]
    template_5tokenized = tokenizer(template_5, add_special_tokens=False)["input_ids"]
    template_6tokenized = tokenizer(template_6, add_special_tokens=False)["input_ids"]
    template_7tokenized = tokenizer(template_7, add_special_tokens=False)["input_ids"]

################################## llama3.1 template ##########################################################################
else:
    template_1 = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Act as an impartial judge and evaluate the quality of responses A and B to the user question. "
        "Choose the response that better follows the user’s instructions, considering factors such as helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail. Be aware that the prompt and responses may be "
        "incomplete due to input length limitations. The evaluation should not be influenced by position biases, "
        "presentation order, or response lengths. Do not favor specific assistant names."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    )
    template_2 = "Question:"
    template_3 = "\n"
    template_4 = "Response A:"
    template_5 = "\n"
    template_6 = "Response B:"
    template_7 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    template_1tokenized = tokenizer(template_1, add_special_tokens=False)["input_ids"]
    template_2tokenized = tokenizer(template_2, add_special_tokens=False)["input_ids"]
    template_3tokenized = tokenizer(template_3, add_special_tokens=False)["input_ids"]
    template_4tokenized = tokenizer(template_4, add_special_tokens=False)["input_ids"]
    template_5tokenized = tokenizer(template_5, add_special_tokens=False)["input_ids"]
    template_6tokenized = tokenizer(template_6, add_special_tokens=False)["input_ids"]
    template_7tokenized = tokenizer(template_7, add_special_tokens=False)["input_ids"]

######################################################################################################################################




### tokenize data
def tokenize_shape(prompt, response_a, response_b, template_1tokenized, template_2tokenized, template_3tokenized, 
                   template_4tokenized, template_5tokenized, template_6tokenized, template_7tokenized, max_length, max_prompt_length):
    p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    a = tokenizer(response_a, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    b = tokenizer(response_b, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

    tokenized = {"input_ids": [], "attention_mask": []}
    for _p, _a, _b in zip(p, a, b):  # 同步遍历 prompt, response_a 和 response_b
        if len(_p) > max_prompt_length:
            _p = _p[-max_prompt_length:]
        len_max = len(template_1tokenized) + len(template_2tokenized) + len(template_3tokenized) + \
                  len(template_4tokenized) + len(template_5tokenized) + len(template_6tokenized) + len(template_7tokenized)
        rl = (max_length - len(_p) - len_max) // 2
        input_ids = [tokenizer.bos_token_id] + template_1tokenized + template_2tokenized + _p + template_3tokenized + \
                    template_4tokenized + _a[-rl:] + template_5tokenized + template_6tokenized + _b[-rl:] + \
                    template_7tokenized + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        tokenized["input_ids"].append(input_ids)
        tokenized["attention_mask"].append(attention_mask)
    return tokenized

def tokenize(
    tokenizer, prompt, response_a, response_b,template_1tokenized,template_2tokenized,template_3tokenized,template_4tokenized,template_5tokenized,template_6tokenized,template_7tokenized, max_length,max_prompt_length
):
    prompt = [ p for p in prompt]
    response_a = [ r_a for r_a in response_a]
    response_b = [ r_b for r_b in response_b]
    tokenized = tokenize_shape(prompt, response_a, response_b,template_1tokenized,template_2tokenized,template_3tokenized,template_4tokenized,template_5tokenized,template_6tokenized,template_7tokenized,max_length,max_prompt_length)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    return input_ids, attention_mask

############################### preprocess data ###########################################
train["input_ids"], train["attention_mask"] = tokenize(tokenizer, train["prompt"], train["response_a"], train["response_b"],template_1tokenized,template_2tokenized,template_3tokenized,template_4tokenized,template_5tokenized,template_6tokenized,template_7tokenized,MAX_LENGTH,MAX_PROMPT_LENGTH)
train['text'] = train['input_ids'].apply(lambda x: tokenizer.decode(x))



train_reverse = train.copy()

def do(row):
    if row['winner']=='model_a':
        return "B"
    else:
        return "A"

train_reverse['label'] = train_reverse.apply(lambda row: do(row), axis=1)
print(train_reverse['label'].value_counts())

train_reverse["input_ids"], train_reverse["attention_mask"] = tokenize(tokenizer, train_reverse["prompt"], train_reverse["response_b"], train_reverse["response_a"],template_1tokenized,template_2tokenized,template_3tokenized,template_4tokenized,template_5tokenized,template_6tokenized,template_7tokenized,MAX_LENGTH,MAX_PROMPT_LENGTH)
train_reverse['text'] = train_reverse['input_ids'].apply(lambda x: tokenizer.decode(x))


def do(x):
    if x == "B":
        return 1
    else:
        return 0

train['label'] = train['label'].apply(lambda x: do(x))
train_reverse['label'] = train_reverse['label'].apply(lambda x: do(x))


train = train.sample(frac=1., random_state=2025)
train_reverse = train_reverse.sample(frac=1., random_state=2025)

train_all = pd.concat([train, train_reverse], axis=0)

print(train_all['label'].value_counts())

with open(
        f"../data/processed_data/ut_{save_name}_train.pkl",
        'wb') as f:
    pickle.dump(train_all, f)

with open(
        f"../data/processed_data/ut_{save_name}_dev.pkl",
        'wb') as f:
    pickle.dump(train_all.sample(n=100), f)


