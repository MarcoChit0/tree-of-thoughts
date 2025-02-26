
import openai
import backoff
from tot.models.model import *

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
    
    @classmethod
    def get_available_backends(cls):
        return ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]

    @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
    def completions_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def __call__(self, prompt, max_tokens=1000, n=1) -> list:
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
        pricing = {
            "gpt-4": {"completion_tokens": 0.06, "prompt_tokens": 0.03},
            "gpt-3.5-turbo": {"completion_tokens": 0.002, "prompt_tokens": 0.0015},
            "gpt-4o": {"completion_tokens": 0.00250, "prompt_tokens": 0.01}
        }
        cost = self.completion_tokens / 1000 * pricing[self.backend]["completion_tokens"] + self.prompt_tokens / 1000 * pricing[self.backend]["prompt_tokens"]
        return {"completion_tokens": self.completion_tokens, "estimated_prompt_tokens": self.estimated_prompt_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}


register_model(GPTModel)