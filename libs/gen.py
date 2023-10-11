import os
import requests
import json

prompt = """
私はゲームを開発しています。
ゲームではモンスターが登場します。
このモンスター特徴は `--insert--` です。
モンスターの名前は """

with open("out.txt", "r") as f:
    for line in f.readlines():
        file_name, text = line.split(",")
        file_name = file_name.strip()
        text = text.strip()

        # print("prompt", prompt.replace("--insert--", text))

        res = requests.post(
            url="http://localhost:8080/completion",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "prompt": prompt.replace("--insert--", text),
                    "stop": ["\n"],
                    # "top_p": 0.2,
                    "temperature": 2,
                }
            ),
        )

        content = res.json()["content"].strip()
        print(f"{file_name},{text},{content}")

        # break
