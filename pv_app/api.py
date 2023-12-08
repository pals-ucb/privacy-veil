'''
   Privacy Veil: Privacy guarantees research on Large Language Models
   APIs
'''
import functools
from flask import Blueprint, g, request, session, url_for
from pv_app.model import get_model
from pv_app.model_loader import alpaca_query, alpaca_query_fast, alpaca_query_with_genconfig

bp = Blueprint('apis', __name__, url_prefix='/privacy-veil/api')

# Hello World API
@bp.route('/hello')
def pv_hello():
    return 'Privacy Veil: Hello, World!\n'

# Alpaca Query for single input_instruction
@bp.route('/alpaca-query', methods=('GET', 'POST'))
def pv_alpaca_query():
    (model, tokenizer) = get_model()
    data = request.get_json()
    print(f'alapac-query: {data}')
    response = alpaca_query(data['input'], model, tokenizer)
    print(f'returning query response: {response}')
    return response

# Alpaca Query fast, expect response in 5 seconds
@bp.route('/alpaca-query-fast', methods=('GET', 'POST'))
def pv_alpaca_query_fast():
    (model, tokenizer) = get_model()
    data = request.get_json()
    print(f'alapac-query-fast: {data}')
    response = alpaca_query_fast(data['input'], model, tokenizer)
    print(f'returning query response: {response}')
    return response

# Alpaca Query with generator configuration
@bp.route('/alpaca-query-with-genconfig', methods=('GET', 'POST'))
def pv_alpaca_query_with_genconfig():
    (model, tokenizer) = get_model()
    data = request.get_json()
    print(f'alapac-query-with-genconfig: {data}')
    response = alpaca_query_with_genconfig(data['input'], data['genconfig'], model, tokenizer)
    print(f'returning query response: {response}')
    return response

