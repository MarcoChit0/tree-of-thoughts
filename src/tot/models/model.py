import os
from tot.tasks.task import Task
from tot.registry import *
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tiktoken
import torch

MAX_NUMBER_CANDIATES_SAMPLED_SINGLE_STEP = 20

class Model(RegistredClass):
    def __init__(self, backend=None, temperature=0.7):
        self.backend = backend
        self.temperature = temperature

    def __call__(self, prompt, max_tokens=1000, n=1):
        raise NotImplementedError("generate method must be implemented in derived classes")
    
    def get_usage(self):
        raise NotImplementedError("get_usage method must be implemented in derived classes")

    @classmethod
    def get_available_backends(cls):
        raise NotImplementedError("get_available_backends method must be implemented in derived classes")

def add_instruct(models):
    return [f"{model}-Instruct" for model in models] + models

def register_model(model_class:Model):
    for backend in model_class.get_available_backends():
        register(MODEL_REGISTRY, model_class, backend)