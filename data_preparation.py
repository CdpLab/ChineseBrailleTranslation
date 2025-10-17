import os
import json
import random

# ===== Paths (relative) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "source_data")
OUT_DIR = os.path.join(BASE_DIR, "Parallel Corpus")
BACKUP_DIR = os.path.join(BASE_DIR, "original_data")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

ZH_PATH = os.path.join(SRC_DIR, "chinese.txt.txt")
BR_PATH = os.path.join(SRC_DIR, "braille.txt")

# ===== Valid Braille chars =====
VALID_BRAILLE = set(
    "⠂⠆⠒⠲⠢⠖⠶⠦⠔⠴⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵"
    "⠮⠐⠼⠫⠩⠯⠄⠷⠾⠡⠬⠠⠤⠨⠌⠆⠰⠣⠿⠜⠹⠈⠪⠳⠻⠘⠸"
)

# ===== Utils =====
def is_valid_braille(text: str) -> bool:
    return all(ch == " " or ch in VALID_BRAILLE for ch in text)

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f if line.strip()]

# ===== Load & align =====
zh_lines = load_lines(ZH_PATH)
br_lines = load_lines(BR_PATH)
n = min(len(zh_lines), len(br_lines))
zh_lines, br_lines = zh_lines[:n], br_lines[:n]
print(f"[Align] {n} pairs")

# ===== Clean =====
pairs = []
drop_invalid = drop_short = drop_dup = 0
seen = set()

for zh, br in zip(zh_lines, br_lines):
    zh, br = zh.strip(), br.strip()  # keep spaces inside Braille

    if not is_valid_braille(br):
        drop_invalid += 1
        continue
    if len(br.replace(" ", "")) < len(zh) * 2:
        drop_short += 1
        continue
    pair = (zh, br)
    if pair in seen:
        drop_dup += 1
        continue
    seen.add(pair)
    pairs.append(pair)

print(f"[Clean] keep={len(pairs)}, invalid={drop_invalid}, short={drop_short}, dup={drop_dup}")

# ===== Build dataset =====
data = [{"input_text": zh, "output_text": br} for zh, br in pairs]
random.shuffle(data)
total = len(data)
train_end, val_end = int(total * 0.8), int(total * 0.9)

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

# ===== Save =====
def save_json(data, name):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Save] {path} ({len(data)})")

for name, subset in splits.items():
    save_json(subset, name)

backup = os.path.join(BACKUP_DIR, "cleaned_pairs.txt")
with open(backup, "w", encoding="utf-8") as f:
    for zh, br in pairs:
        f.write(f"{zh}\t{br}\n")
print(f"[Backup] {backup}")

print("✅ Done.")
