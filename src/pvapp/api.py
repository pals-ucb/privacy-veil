'''
   Privacy Veil: Privacy guarantees research on Large Language Models
   APIs
'''
import functools
from flask import Blueprint, g, request, session, url_for
from pvapp.model import get_model
from pvapp.model_loader import alpaca_query_single

bp = Blueprint('apis', __name__, url_prefix='/privacy-veil/api')

# Hello World API
@bp.route('/hello')
def pv_hello():
    return 'Privacy Veil: Hello, World!\n'

# Alpaca Query for single input_str
@bp.route('/alpaca-query-single', methods=('GET', 'POST'))
def pv_alpaca_query_single():
    (model, tokenizer) = get_model()
    data = request.get_json()
    print(f'alapac-query-single: {data}')
    response = alpaca_query_single(data['input'], model, tokenizer)
    return 'Privacy Veil: Hello, World!\n'

