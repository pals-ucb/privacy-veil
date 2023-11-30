'''
   Privacy Veil: Privacy guarantees research on Large Language Models
'''
import os
from flask import Flask
import requests

def pv_init_app(app_name, test_config):
    # create and configure the app
    app = Flask(app_name, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'privacy-veil.sqlite'),
        MODEL_PATH=os.path.join(app.instance_path, 'models/peft-tune-llama-2-7b-chat-hf-credit-card-fraud-v2'),
        DEVICE='mps')

    if test_config is None:
        # load the instance config
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Otherwise load the test config 
        app.config.from_mapping(test_config)

    app.config.from_prefixed_env()
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Load the model support infra
    from . import model
    model.init_app(app)

    # Load the api support infra
    from . import api
    app.register_blueprint(api.bp)
    return app

def create_app(test_config=None):
    return pv_init_app(__name__, test_config)

