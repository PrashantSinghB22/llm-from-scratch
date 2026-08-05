from pathlib import Path

from tokenizer.character_tokenizer import (
    read_text_file,
    build_vocabulary,
    build_mappings,
)

data_path = Path("data") / "wizard_of_oz.txt"

text = read_text_file(data_path)

chars = build_vocabulary(text)

stoi, itos = build_mappings(chars)

print("Vocabulary Size:", len(chars))

print()

print("First 10 character mappings")

for char in chars[:10]:
    print(repr(char), "→", stoi[char])