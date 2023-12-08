'''
   Privacy Veil: Privacy guarantees research on Large Language Models
   Model Loader
'''
import torch
import transformers
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from flask import current_app

def load_model(model_name):
    device = current_app.config['DEVICE']
    print(f'Starting to load the model {model_name}')
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
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

def clean_response(input_str, response):
    # I think this may not be good to clean
    # the response.
    # Let us experiment a bit more and then see.
    '''
    ret = []
    for resp in response:
        if len(resp) > len(input_str):
            resp = resp[len(input_str)+1:]
        resp = resp.replace('\\n', '\n')
        ret.append(resp)
    return ret
    '''
    return response

def alpaca_query(input_str, model, tokenizer):
    device = current_app.config['DEVICE']
    tokens = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in tokens.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=1024
        )
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    return clean_response(input_str, result)

def alpaca_query_with_genconfig(input_str, genconfig, model, tokenizer):
    device = current_app.config['DEVICE']
    tokens = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in tokens.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **genconfig
        )
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    return clean_response(input_str, result)

def alpaca_query_fast(input_str, model, tokenizer):
    device = current_app.config['DEVICE']
    tokens = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in tokens.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
            min_new_tokens=10,
            max_new_tokens=80,  
            max_time = 5,
            do_sample = True,
            temperature = 0.5,
            top_k = 3,
            repetition_penalty = 2.0
        )
    result = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    print(result)
    print(type(result))
    return clean_response(input_str, result)

