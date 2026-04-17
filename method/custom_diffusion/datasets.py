import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_image_rgb(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0
    return torch.from_numpy(image).permute(2, 0, 1)


def _replace_token(text: str, target_token: str, modifier_token: str) -> str:
    if not text:
        return ""
    return str(text).replace(target_token, modifier_token)


@dataclass
class SERCustomDiffusionDataConfig:
    train_json: str
    image_size: int = 512
    target_token: str = "[V]"
    modifier_token: str = "<new1>"


class SERCustomDiffusionDataset(Dataset):
    """Samples a random history item per SER sample."""

    def __init__(self, cfg: SERCustomDiffusionDataConfig):
        super().__init__()
        self.cfg = cfg
        self.data: List[Dict] = _load_json(cfg.train_json)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        history_items = sample.get("history_items_info", [])
        if not history_items:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        for _ in range(10):
            hist = random.choice(history_items)
            image_path = hist.get("image_path")
            if not image_path or not os.path.exists(image_path):
                continue

            prompt = hist.get("masked_caption") or hist.get("caption") or ""
            prompt = _replace_token(prompt, self.cfg.target_token, self.cfg.modifier_token)
            if not prompt.strip():
                continue

            pixel_values = _read_image_rgb(image_path, self.cfg.image_size)
            return {"pixel_values": pixel_values, "prompt": prompt}

        return self.__getitem__(random.randint(0, len(self.data) - 1))


@dataclass
class POGCustomDiffusionDataConfig:
    train_json: str
    masked_captions_file: str
    original_captions_file: str
    image_size: int = 512
    target_token: str = "[V]"
    modifier_token: str = "<new1>"


class POGCustomDiffusionDataset(Dataset):
    """Samples a random history item and uses masked caption by item_id when available."""

    def __init__(self, cfg: POGCustomDiffusionDataConfig):
        super().__init__()
        self.cfg = cfg
        self.data: List[Dict] = _load_json(cfg.train_json)
        self.masked: Dict[str, str] = _load_json(cfg.masked_captions_file) if os.path.exists(cfg.masked_captions_file) else {}
        self.original: Dict[str, str] = _load_json(cfg.original_captions_file) if os.path.exists(cfg.original_captions_file) else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        history_items = sample.get("history_items_info", [])
        if not history_items:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        for _ in range(20):
            hist = random.choice(history_items)
            image_path = hist.get("image_path")
            item_id = hist.get("item_id", "")
            if not image_path or not os.path.exists(image_path):
                continue

            masked_caption = self.masked.get(item_id, "")
            if not masked_caption:
                masked_caption = hist.get("caption", "")
            if not masked_caption:
                continue

            prompt = _replace_token(masked_caption, self.cfg.target_token, self.cfg.modifier_token)
            if self.cfg.modifier_token not in prompt:
                continue

            pixel_values = _read_image_rgb(image_path, self.cfg.image_size)
            return {"pixel_values": pixel_values, "prompt": prompt}

        return self.__getitem__(random.randint(0, len(self.data) - 1))


@dataclass
class FlickrAESCustomDiffusionDataConfig:
    train_json: str
    masked_captions_file: str
    original_captions_file: str
    image_size: int = 512
    target_token: str = "[V]"
    modifier_token: str = "<new1>"


class FlickrAESCustomDiffusionDataset(Dataset):
    """Samples a random item from interaction_sequence and uses masked caption dict keyed by item_id (strip .jpg)."""

    def __init__(self, cfg: FlickrAESCustomDiffusionDataConfig):
        super().__init__()
        self.cfg = cfg
        self.data: List[Dict] = _load_json(cfg.train_json)
        self.masked: Dict[str, str] = _load_json(cfg.masked_captions_file) if os.path.exists(cfg.masked_captions_file) else {}
        self.original: Dict[str, str] = _load_json(cfg.original_captions_file) if os.path.exists(cfg.original_captions_file) else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        seq = sample.get("interaction_sequence", [])
        if not seq:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        for _ in range(50):
            item = random.choice(seq)
            image_path = item.get("image_path")
            item_id = item.get("item_id", "")
            if not image_path or not os.path.exists(image_path):
                continue

            key = item_id[:-4] if item_id.endswith(".jpg") else item_id
            masked_caption = self.masked.get(key, "")
            if not masked_caption:
                continue

            prompt = _replace_token(masked_caption, self.cfg.target_token, self.cfg.modifier_token)
            if self.cfg.modifier_token not in prompt:
                continue

            pixel_values = _read_image_rgb(image_path, self.cfg.image_size)
            return {"pixel_values": pixel_values, "prompt": prompt}

        return self.__getitem__(random.randint(0, len(self.data) - 1))
