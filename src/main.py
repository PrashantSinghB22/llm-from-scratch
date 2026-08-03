from tokenizer.character_tokenizer import read_text_file, build_vocabulary
from pathlib import Path

data_path = Path("data") / "wizard_of_oz.txt"

text = read_text_file(data_path)

chars = build_vocabulary(text)
print(f"Vocabulary Size: {len(chars)}")
print(chars)