import os

folder_path = "/data2/xintong/LibriTTS/train-clean-100"

wav_files = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, file))

import json

with open("/home/junchuan/EnCodec_Trainer/LibriTTS_meta.json", "w", encoding="utf-8") as f:
    json.dump(wav_files, f, ensure_ascii=False, indent=4)
