import json
from tensorflow.keras.models import model_from_json
from .io import save_json, load_json

def save_architecture(fname, model):
    config = model.to_json()
    save_json(fname, config)
    return

def load_architecture(fname):
    config = load_json(fname)
    model = model_from_json(config)
    return model
