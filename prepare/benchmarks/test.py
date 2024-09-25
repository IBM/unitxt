from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x22B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
