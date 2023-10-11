from typing import Any
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration  # type: ignore
import torch
import os


# ----- setup -----
device: str = "cuda" if torch.cuda.is_available() else "cpu"
model_name: str = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name, cache_dir="cache")
model: Any = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    cache_dir="cache",
)
model.to(device)


def predict(path: str):
    image = Image.open(path, "r").convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    ids = model.generate(**inputs)
    return processor.batch_decode(ids, skip_special_tokens=True, fp16=False)[0].strip()


def get_file_path(directory_path: str):
    files: list[str] = []
    for dir in os.listdir(directory_path):
        _dir = os.path.join(directory_path, dir)
        for file in os.listdir(_dir):
            files.append(os.path.join(_dir, file))
    return files


files: list[str] = []
for file in get_file_path("images"):
    files.append(file)

for f in files:
    print(f"{f},{predict(f)}")
