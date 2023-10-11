from typing import Any
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration  # type: ignore
import torch
from ray.util import ActorPool
import ray
import os


ray.init(object_store_memory=10**9)

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

# object store へ個別に格納している
P = ray.put(processor)
M = ray.put(model)

# ↑ Actor は複数作成される場合がある為、Actor は複数である場合、このように分けるべき
# ↓ Actor は今回単一の為、put ではなく、内部にしなければ、Actor 分メモリが必要になるのでは？
# https://docs.ray.io/en/releases-1.11.0/_images/memory.svg


# メモリ使用量がどの程度の想定であるか伝えておく（利用自体が制限されるものではない） = 15GB
# https://docs.ray.io/en/releases-1.11.0/ray-core/memory-management.html
@ray.remote(memory=15000 * 1024 * 1024)
class Actor:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def predict(self, path: str):
        print(f"Processing,{path}")
        image = Image.open(path, "r").convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        ids = self.model.generate(**inputs)
        return self.processor.batch_decode(ids, skip_special_tokens=True, fp16=False)[
            0
        ].strip()


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

actors = [Actor.remote(P, M) for _ in range(1)]
print("actors", len(actors))
pool = ActorPool(actors).map(lambda a, f: a.predict.remote(f), files)
print("1. Predicting...")
print(list(pool))

# actor = Actor.options(max_concurrency=1).remote(P, M)

# results = asyncio.gather(
#     actor.predict.remote(files[0]),
#     actor.predict.remote(files[0]),
#     actor.predict.remote(files[0]),
# )
