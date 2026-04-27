import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image_rgb(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0
    return torch.from_numpy(image).permute(2, 0, 1)


def replace_target_token(text: str, target_token: str, instance_token: str) -> str:
    if not text:
        return ""
    return str(text).replace(target_token, instance_token)


class HistoryItemsDreamBoothDataset(Dataset):
    def __init__(self, *, train_json: str, image_size: int = 512, target_token: str = "[V]", instance_token: str = "sks"):
        super().__init__()
        self.train_json = train_json
        self.image_size = image_size
        self.target_token = target_token
        self.instance_token = instance_token
        self.data: List[Dict] = load_json(train_json)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        history_items = sample.get("history_items_info", [])
        if not history_items:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        for _ in range(20):
            hist = random.choice(history_items)
            image_path = str(hist.get("image_path") or "")
            if not image_path or not os.path.exists(image_path):
                continue

            prompt = str(hist.get("masked_caption") or hist.get("caption") or "")
            prompt = replace_target_token(prompt, self.target_token, self.instance_token)
            if not prompt.strip():
                continue

            pixel_values = read_image_rgb(image_path, self.image_size)
            return {"pixel_values": pixel_values, "prompt": prompt}

        return self.__getitem__(random.randint(0, len(self.data) - 1))
