from models import *

class HuggingFaceModel(Model):
    def __init__(self, backend, available_backends, path, temperature=0.7):
        super().__init__(backend, available_backends, temperature)
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

    
class LlamaModel(HuggingFaceModel):
    def __init__(self, backend="llama", temperature=0.7):
        models_with_instruct = ["llama-3.2-1B", "llama-3.2-3B", "llama-3.1-8B", "llama-3.1-70B"]
        models_without_instruct = []
        models = add_instruct(models_with_instruct) + models_without_instruct
        super().__init__(backend, models, "meta-llama", temperature)

register(LlamaModel, MODEL_REGISTRY)

        

class QwenModel(HuggingFaceModel):
    def __init__(self, backend="Qwen2.5-0.5B", temperature=0.7):
        models_with_instruct = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
        models_without_instruct = []
        models = add_instruct(models_with_instruct) + models_without_instruct
        super().__init__(backend, models, "Qwen", temperature)

register(QwenModel, MODEL_REGISTRY)