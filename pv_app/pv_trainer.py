#!/usr/bin/env python
import argparse
import os
import sys
import re
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, BloomForCausalLM, BloomTokenizerFast
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training

# Some sane default values. Touch them, if you know what they mean.
MICRO_BATCH_SIZE = 4  
BATCH_SIZE = 128 
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4  
DEVICE_MAP = 'auto'

def pv_find_target_modules_for_LoRA(model):
    model_modules = str(model.modules)
    re_expr = r'\((\w+)\): Linear'
    ll_names = re.findall(re_expr, model_modules)
    ns = []
    for n in ll_names:
        print(f'Linear layer in model: {n}')
        ns.append(n)
    # Drop duplicates
    return list(set(ns))

def pv_tokenize(r):
    inst_tokens = tokenizer(r['Instructions'] + r['Input'], truncation=True, 
                              max_length=args.cutoff_len + 1)
    inst_len = len(inst_tokens['input_ids']) - 1
    all_tokens = tokenizer(r['Instructions'] + r['Input'] + r['Output'], truncation=True, 
                           max_length=args.cutoff_len + 1, padding="max_length")               
    all_len = len(all_tokens['input_ids']) - 1
    return {
        "input_ids": all_tokens['input_ids'][:-1],
        "labels": [-100] * inst_len + all_tokens['input_ids'][inst_len:all_len],
        "attention_mask": [1] * all_len,
    }

def pv_initialize_training(args):
    project_dir = f'./{args.project_name}'
    try:
        os.makedirs(project_dir)
    except FileExistsError:
        pass
    if args.llama:
        model = LlamaForCausalLM.from_pretrained(args.base_model,
            load_in_8bit=args.load_in_8bit,
            device_map=DEVICE_MAP)
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model, add_eos_token=True)
    elif args.bloomz:
        model = BloomForCausalLM.from_pretrained(args.base_model,
            load_in_8bit=args.load_in_8bit,
            device_map=DEVICE_MAP)
        tokenizer = BloomTokenizerFast.from_pretrained(args.base_model, model_max_length=args.cutoff_len)
    else:
        print(f'Must select on model type for training.')
        sys.exit(-1)
        
    model = prepare_model_for_int8_training(model)
    target_modules = pv_find_target_modules_for_LoRA(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0 
    return project_dir, model, tokenizer

def pv_prepare_dataset(args):
    ds = load_dataset("csv", data_files={"train": [args.data_file]})
    ds = ds['train'].select_columns(['Instructions', 'Input', 'Output'])
    split_ds = ds.train_test_split(test_size=args.split_percent/100.0)
    train_ds = split_ds["train"].map(pv_tokenize)
    test_ds  = split_ds["test"].map(pv_tokenize)
    train_ds = train_ds.select_columns(['input_ids', 'labels', 'attention_mask'])
    test_ds  = test_ds.select_columns(['input_ids', 'labels', 'attention_mask'])
    print(f'Prepared and got the tensors for training')
    return train_ds, test_ds

def pv_start_training(args, project_dir, model, tokenizer, train_ds, test_ds):
    model.is_parallelizable = args.use_gpu
    model.model_parallel = args.use_gpu

    train_args = TrainingArguments(
        num_train_epochs=args.num_epochs,
        learning_rate=LEARNING_RATE,
        fp16=args.fp16,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=project_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to='none')

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print(f'Created trainer object')
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    if torch.__version__ >= "2":
        print(f'Torch version: {torch.__version__}')
        model = torch.compile(model)
    print(f'Start the training process')
    trainer.train()
    model.save_pretrained(project_dir)
    tokenizer.save_pretrained(project_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Privacy Veil LLM trainer for Llama/Bloomz models')
    parser.add_argument('--project_name', default='pv_output_dir', type=str, help='name of the project')
    parser.add_argument('--llama', action='store_true', default=False,  help='Train meta-llama')
    parser.add_argument('--bloomz', action='store_true', default=False, help='Train bloomz')
    parser.add_argument('--use_gpu', action='store_true',default=False,  help='Train using the GPU.')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='Use 8bit integers.')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use 1/2 precision floating point.')
    parser.add_argument('--data_file', type=str, default='./train.csv', help='Training .csv file.')
    parser.add_argument('--base_model', default="meta-llama/llama-2-7b-chat-hf", type=str, help='name of the model as shown in huggingface model card')
    parser.add_argument('--num_epochs', default=3, type=int, help='Number of epochs used for training')
    parser.add_argument('--cutoff_len', default=512, type=int, help='Maximum cut off len for train data')
    parser.add_argument('--split_percent', default=10, type=int, help='Percentage of data used for testing and rest for training.')
    args = parser.parse_args()

    project_dir, model, tokenizer = pv_initialize_training(args)
    print(model.print_trainable_parameters())
    train_ds, test_ds = pv_prepare_dataset(args)
    pv_start_training(args, project_dir, model, tokenizer, train_ds, test_ds)
    print(f'Training completed')
