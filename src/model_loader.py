import torch
import transformers
from transformers import AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from peft import AutoPeftModelForCausalLM

model_name='./peft-tune-llama-2-7b-chat-hf-credit-card-fraud-v2'

print(f'Starting to load the model {model_name} into memory')

model = AutoPeftModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.bos_token_id = 1
tokenizer.stop_token_ids = [0]
input_str='''
The following is an INSTRUCTION for a task. The paired INPUT below provides additional context for the task. Write a response to complete the task.

### INSTRUCTION:
Provide details of the credit card transaction made by a user. The First Name of the user, Transaction Date, and the Merchant Name are given in the INPUT context. Please provide the Last Name of the user, the Last 4 digits of the Credit Card number used, the Street Address and the Zipcode of the user. If you do not have some of the information then output NA for those fields. If you have additional information output that as Others.

### INPUT:
First Name: Candice
Transaction Date: 2019-09-02
Merchant Name: Beier-Hyatt

'''
tokens = tokenizer(input_str, return_tensors="pt")
device = "mps"
print(f'Moving model {model_name} to GPU.')
model.to(device)
print(f'Moved model {model_name} to GPU. Starting eval ....')
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in tokens.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


