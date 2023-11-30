'''
   Privacy Veil: Model loader interface
'''
import click
from . import model_loader
from flask import current_app, g

global current_model

def get_model():
    global current_model
    if not current_model:
        current_model = model_loader.load_model(current_app.config['MODEL_PATH'])
    return current_model

def close_model(e=None):
    #model = g.pop('model', None)
    #if model is not None:
    #    model_loader.unload_model(current_app.config['MODEL_PATH'])
    pass

def init_model():
    get_model()
    
@click.command('load-model')
def load_model_command():
    """Clear the existing model and load the new model"""
    global current_model
    current_model = None
    init_model()
    click.echo('Initialized and loaded the new model')

def init_app(app):
    global current_model
    current_model = None
    app.teardown_appcontext(close_model)
    app.cli.add_command(load_model_command)

