from tot.models.model import *

class HuggingFaceModel(Model):
    def __init__(self, backend,  path, temperature=0.7):
        super().__init__(backend, temperature)
        self.path = path
        self.generated_tokens = 0
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        if self.token is None:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.path}/{self.backend}", token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.path}/{self.backend}", token=self.token)

    def __call__(self, prompt, max_tokens=1000, n=1) -> list[str]:
        messages = [{"role": "user", "content": prompt}]

        for m in messages:
            print(f"Message:\n{m}\n")

        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        outputs = generator(messages, max_new_tokens=max_tokens, num_return_sequences=n, do_sample=True, temperature=self.temperature)

        cleaned_outputs = []
        for output in outputs:
            i = 0
            while i < len(output["generated_text"]) and output["generated_text"][i] in messages:
                i += 1
            
            for j in range(i, len(output["generated_text"])):
                cleaned_outputs.append(output["generated_text"][j]['content'])

        self.generated_tokens += sum(len(self.tokenizer.encode(output)) for output in cleaned_outputs)

        for output in cleaned_outputs:
            print(f"Output:\n{output}\n")

        return cleaned_outputs

    def get_usage(self) -> dict:
        return {"generated_tokens": self.generated_tokens}
    

    
class LlamaModel(HuggingFaceModel):
    def __init__(self, backend="llama-3.2-1B", temperature=0.7):
        super().__init__(backend, "meta-llama", temperature)
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.path}/{self.backend}", token=self.token, torch_dtype=torch.float16)
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @classmethod
    def get_available_backends(cls) -> list[str]:
        models_with_instruct = ["llama-3.2-1B", "llama-3.2-3B", "llama-3.1-8B", "llama-3.1-70B"]
        models_without_instruct = []
        models = add_instruct(models_with_instruct) + models_without_instruct
        return models

register_model(LlamaModel)
        

class QwenModel(HuggingFaceModel):
    def __init__(self, backend="Qwen2.5-0.5B", temperature=0.7):
        super().__init__(backend, "Qwen", temperature)
    
    @classmethod
    def get_available_backends(cls) -> list[str]:
        models_with_instruct = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
        models_without_instruct = []
        models = add_instruct(models_with_instruct) + models_without_instruct
        return models

register_model(QwenModel)