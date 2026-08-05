from pathlib import Path

from tokenizer.character_tokenizer import (
    read_text_file,
    build_vocabulary,
    build_mappings,
    encode,
)

data_path = Path("data") / "wizard_of_oz.txt"

text = read_text_file(data_path)

chars = build_vocabulary(text)

stoi, itos = build_mappings(chars)

sample = "Hello"

encoded = encode(sample, stoi)

print(sample)

print(encoded)