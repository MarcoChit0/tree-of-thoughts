from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is the capital of France?"
# chat message template
messages = [
    {"role": "system", "content": "You are a chatbot assisting a user with their geography questions."},
    {"role": "user", "content": prompt},
]
# apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# tokenize and send tokens to model
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generate response
generated_ids = model.generate(
    **model_inputs
)
# remove input tokens
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
# decode response
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)