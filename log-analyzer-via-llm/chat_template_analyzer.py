from transformers import AutoTokenizer

model_name = "../qwen25-3b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer.chat_template)