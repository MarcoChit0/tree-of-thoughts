
from google import genai
from google.genai import types
from tot.models.model import *

class GeminiModel(Model):

    def __init__(self, backend="gemini-1.5-flash-8b", temperature=0.7):
        super().__init__(backend, temperature)
        self.generated_tokens = 0
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=self.api_key)
        self.response_tokens = 0 
        self.prompt_tokens = 0
        self.total_tokens = 0
        
    @classmethod
    def get_available_backends(cls) -> list[str]:
        return ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

    def __call__(self, prompt, max_tokens=1000, n=1) -> list[str]:
        messages = [prompt]
        res = self.client.models.generate_content(
            model=self.backend,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                candidate_count=3,
                max_output_tokens=max_tokens
            )
        )

        outputs = []
        for candidate in res.candidates:
            text = ""
            for part in candidate.content.parts:
                text += part.text
            outputs.append(text)

        self.response_tokens += res.usage_metadata.candidates_token_count
        self.prompt_tokens += res.usage_metadata.prompt_token_count
        self.total_tokens += res.usage_metadata.total_token_count

        return outputs

    def get_usage(self) -> dict:
        return {
            "response_tokens": self.response_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens
        }

register_model(GeminiModel)