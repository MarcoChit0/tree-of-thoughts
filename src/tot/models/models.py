import os
from tot.tasks.task import Task
from tot.registry import register, MODEL_REGISTRY
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tiktoken
import torch

MAX_NUMBER_CANDIATES_SAMPLED_SINGLE_STEP = 20

class Model:
    
    def __init__(self, backend=None, available_backends=None, temperature=0.7):
        self.backend = backend
        self.available_backends = available_backends
        self.temperature = temperature

    def generate(self, prompt, max_tokens=1000, n=1):
        raise NotImplementedError("generate method must be implemented in derived classes")
    
    def get_usage(self):
        raise NotImplementedError("get_usage method must be implemented in derived classes")

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.backend)
        except Exception:
            # Fallback to a default encoding if the model isn't recognized
            print(f"Model {self.backend} not recognized for tokenization, using cl100k_base encoding")
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_proposals(self, task, x, y): 
        propose_prompt = task.propose_prompt_wrap(x, y)
        print(f"propose_prompt:\n---------------\n{propose_prompt}---------------\n")
        proposals = self.generate(propose_prompt, n=1)[0].split('\n')
        print("proposals:\n---------------\n")
        for proposal in proposals:
            print('#############')
            print(proposal)
            print('#############')
        print('---------------\n')
        
        return [y + _ + '\n' for _ in proposals]

    def get_samples(self, task, x, y, n_generate_sample, prompt_sample):
        if prompt_sample == 'standard':
            prompt = task.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = task.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        samples = self.generate(prompt, n=n_generate_sample)
        return [y + _ for _ in samples]

def add_instruct(models):
    return [f"{model}-Instruct" for model in models] + models