import os
import json
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Chinese_braille_data", "Parallel Corpus")
MODEL_DIR = os.path.join(BASE_DIR, "down_model")

def directory_exists_and_not_empty(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0

if not directory_exists_and_not_empty(MODEL_DIR):
    print(f"Directory {MODEL_DIR} does not exist or is empty. Downloading mT5-small...")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model and tokenizer saved to {MODEL_DIR}.")
else:
    print(f"Model directory {MODEL_DIR} already exists. Skipping download.")

unique_chars = set()
for filename in ["train.json", "val.json", "test.json"]:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for sample in data:
            text = sample.get("output_text", "")
            unique_chars.update(text)

print(f"Detected {len(unique_chars)} unique characters from dataset.")

tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)

added_tokens = tokenizer.add_tokens(sorted(unique_chars))
print(f"Added {added_tokens} new tokens to tokenizer.")

if added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)
print(f"Updated model and tokenizer saved to {MODEL_DIR}.")
