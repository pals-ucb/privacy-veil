import torch
import transformers
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from flask import current_app

def load_model(model_name):
    device = current_app.config['DEVICE']
    print(f'Starting to load the model {model_name}')
    model = AutoPeftModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1
    tokenizer.stop_token_ids = [0]
    print(f'Moving model {model_name} to GPU {device}.')
    model.to(device)
    print(f'Moved model {model_name} to GPU. Starting eval ....')
    return (model, tokenizer)

def unload_model(model_name):
    #print(f'Unloading model: {model_name}')
    pass

def alpaca_query_single(input_str, model, tokenizer):
    device = current_app.config['DEVICE']
    print(f'Making alpaca query with the given input_str: {input_str}')
    tokens = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in tokens.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    return result

