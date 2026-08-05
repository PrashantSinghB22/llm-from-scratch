from pathlib import Path
def read_text_file(file_path):
  with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

  return text

def build_vocabulary(text):
  chars = sorted(set(text))
  return chars

def build_mappings(chars):
  stoi = {ch: i for i, ch in enumerate(chars)}
  itos = {i: ch for i, ch in enumerate(chars)}

  return stoi, itos