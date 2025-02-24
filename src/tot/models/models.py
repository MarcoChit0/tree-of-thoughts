import os
from tot.tasks.base import Task
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tiktoken
import torch

MAX_NUMBER_CANDIATES_SAMPLED_SINGLE_STEP = 20

class Model:
    
    def __init__(self, backend=None, temperature=0.7):
        self.backend = backend
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


    def get_value(self, task, x, y, n_evaluate_sample, cache_value=True):
        value_prompt = task.value_prompt_wrap(x, y)
        if cache_value and value_prompt in task.value_cache:
            return task.value_cache[value_prompt]
        value_outputs = self.generate(value_prompt, n=n_evaluate_sample)
        value = task.value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            task.value_cache[value_prompt] = value
        return value

    def get_values(self, task, x, ys, n_evaluate_sample, cache_value=True):
        values = []
        local_value_cache = {}
        print("---------------\nvalues:\n---------------\n")
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:    
                print("###########")
                print('y:', y)
                value = self.get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                local_value_cache[y] = value
                print("value: ", value)
                print("###########")
            values.append(value)
        print('---------------\n')
        return values

    def get_votes(self, task, x, ys, n_evaluate_sample):
        vote_prompt = task.vote_prompt_wrap(x, ys)
        vote_outputs = self.generate(vote_prompt, n=n_evaluate_sample)
        values = task.vote_outputs_unwrap(vote_outputs, len(ys))
        return values

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

import openai
import backoff
gpt_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]

class GPTModel(Model):
    def __init__(self, backend="gpt-4", temperature=0.7, stop=None):
        super().__init__(backend, temperature)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.stop = stop
        self.estimated_prompt_tokens = 0

        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise ValueError("OPENAI_API_KEY is not set")
        
        api_base = os.getenv("OPENAI_API_BASE", "")
        if api_base != "":
            print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
            openai.api_base = api_base

        
    @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def generate(self, prompt, max_tokens=1000, n=1) -> list:
        messages = [{"role": "user", "content": prompt}]
        outputs = []
        estimated_prompt_tokens = self.count_tokens(prompt, self.backend)
        self.estimated_prompt_tokens = estimated_prompt_tokens
        while n > 0:
            cnt = min(n, MAX_NUMBER_CANDIATES_SAMPLED_SINGLE_STEP)
            n -= cnt
            res = self.completions_with_backoff(model=self.backend, messages=messages, temperature=self.temperature, max_tokens=max_tokens + estimated_prompt_tokens, n=cnt, stop=self.stop)
            outputs.extend([choice.message.content for choice in res.choices])
            # log completion tokens
            self.completion_tokens += res.usage.completion_tokens
            self.prompt_tokens += res.usage.prompt_tokens
        return outputs

    def get_usage(self):
        if self.backend == "gpt-4":
            cost = self.completion_tokens / 1000 * 0.06 + self.prompt_tokens / 1000 * 0.03
        elif self.backend == "gpt-3.5-turbo":
            cost = self.completion_tokens / 1000 * 0.002 + self.prompt_tokens / 1000 * 0.0015
        elif self.backend == "gpt-4o":
            cost = self.completion_tokens / 1000 * 0.00250 + self.prompt_tokens / 1000 * 0.01
        return {"completion_tokens": self.completion_tokens, "estimated_prompt_tokens": self.estimated_prompt_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}

from google import genai
from google.genai import types

gemini_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

class GeminiModel(Model):

    def __init__(self, backend="gemini-1.5-flash-8b", temperature=0.7):
        super().__init__(backend, temperature)
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt, max_tokens=1000, n=1):
        response = self.client.models.generate_content(
            model=self.backend,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=self.temperature,
                candidate_count=n,
            )
        )
        self.prompt_tokens += self.count_tokens(prompt)
        print(response.candidates)
        candidates_text = [candidate.content.parts[0].text for candidate in response.candidates]
        self.response_tokens += sum(self.count_tokens(candidate) for candidate in candidates_text)
        return candidates_text

    def get_usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.prompt_tokens + self.response_tokens
        }

class HuggingFaceModel(Model):
    def __init__(self, backend, path, temperature=0.7):
        super().__init__(backend, temperature)
        self.path = path
        self.generated_tokens = 0
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if self.token is None:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

    def generate(self, prompt, max_tokens=1000, n=1):
        # Initialize the pipeline
        # Load with authentication
        tokenizer = AutoTokenizer.from_pretrained(f"{self.path}/{self.backend}", token=self.token)
        model = AutoModelForCausalLM.from_pretrained(f"{self.path}/{self.backend}", token=self.token)

        # Create the pipeline without passing use_auth_token again
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # Generate text
        outputs = generator(prompt, max_new_tokens=max_tokens, num_return_sequences=n, do_sample=True, temperature=self.temperature)

        self.generated_tokens += sum(len(output['generated_text']) for output in outputs)

        # Extract and clean the generated text
        cleaned_outputs = []
        for output in outputs:
            generated = output['generated_text']
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            cleaned_outputs.append(generated.strip())

        return cleaned_outputs

    def get_usage(self):
        return {"generated_tokens": self.generated_tokens}
    
llama_models = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.1-70B"]
class LlamaModel(HuggingFaceModel):
    def __init__(self, backend="llama", temperature=0.7):
        super().__init__(backend, "meta-llama", temperature)


qwen_models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
class QwenModel(HuggingFaceModel):
    def __init__(self, backend="Qwen2.5-0.5B", temperature=0.7):
        super().__init__(backend, "Qwen", temperature)

def add_instruct(models):
    return [f"{model}-Instruct" for model in models] + models

def get_available_backends():
    global qwen_models, llama_models, gpt_models, gemini_models
    return add_instruct(qwen_models) + add_instruct(llama_models) + gpt_models + gemini_models

def get_model(args: argparse.Namespace) -> Model:
    global qwen_models, llama_models, gpt_models, gemini_models
    if args.backend in gpt_models:
        return GPTModel(args.backend, args.temperature)
    elif args.backend in add_instruct(qwen_models):
        return QwenModel(args.backend, args.temperature)
    elif args.backend in add_instruct(llama_models):
        return LlamaModel(args.backend, args.temperature)
    elif args.backend in gemini_models:
        return GeminiModel(args.backend, args.temperature)
    else:
        raise ValueError(f"model {args.backend} not recognized")
