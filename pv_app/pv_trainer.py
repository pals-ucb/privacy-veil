#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training

MICRO_BATCH_SIZE = 4  
BATCH_SIZE = 128 
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
CUTOFF_LEN = 512  
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
EPOCHS = 3 
LEARNING_RATE = 1e-4  
TEST_DS = 0.3
USE_GPU = True
DEVICE_MAP = 'auto'
PROJECT_NAME = 'test_train'
PROJECT_DIR  = f'./{PROJECT_NAME}'

try:
    os.makedirs(PROJECT_DIR)
except FileExistsError:
    pass

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map=DEVICE_MAP,
)
tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", add_eos_token=True
)

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0 

def pv_tokenize(r):
    inst_tokens = tokenizer(r['Instructions'] + r['Input'], truncation=True, 
                              max_length=CUTOFF_LEN + 1)
    inst_len = len(inst_tokens['input_ids']) - 1
    all_tokens = tokenizer(r['Instructions'] + r['Input'] + r['Output'], truncation=True, 
                           max_length=CUTOFF_LEN + 1, padding="max_length")               
    all_len = len(all_tokens['input_ids']) - 1
    return {
        "input_ids": all_tokens['input_ids'][:-1],
        "labels": [-100] * inst_len + all_tokens['input_ids'][inst_len:all_len],
        "attention_mask": [1] * all_len,
    }

ds = load_dataset("csv", data_files={"train": ["./train.csv"]})
ds = ds['train'].select_columns(['Instructions', 'Input', 'Output'])
split_ds = ds.train_test_split(test_size=TEST_DS)

train_ds = split_ds["train"].map(pv_tokenize)
test_ds  = split_ds["test"].map(pv_tokenize)
train_ds = train_ds.select_columns(['input_ids', 'labels', 'attention_mask'])
test_ds  = test_ds.select_columns(['input_ids', 'labels', 'attention_mask'])

print(f'Prepared and got the tensors for training')
if USE_GPU:
    model.is_parallelizable = True
    model.model_parallel = True
else:
    model.is_parallelizable = False
    model.model_parallel = False

args = TrainingArguments(
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=PROJECT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True)

#args = args.set_logging(strategy="steps", steps=100, report_to=['none'], level="warning" )

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print(f'Created trainer object')
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2":
    print(f'Torch version: {torch.__version__}')
    model = torch.compile(model)

print(f'Start the training process')
trainer.train()
model.save_pretrained(PROJECT_DIR)

