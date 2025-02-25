
from google import genai
from google.genai import types
from models import *

class GeminiModel(Model):

    def __init__(self, backend="gemini-1.5-flash-8b", temperature=0.7):
        models = ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
        super().__init__(backend, models, temperature)
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

register(GeminiModel, MODEL_REGISTRY)