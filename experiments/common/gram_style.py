"""VGG-19 Gram-matrix style distance between two images (low-level style transfer setup)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def _gram(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


class VGGStyleEncoder(nn.Module):
    """Slice VGG19 features up to relu4_1 (0:21 layers in torchvision ordering)."""

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.net = vgg.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.slice_ids: List[Tuple[int, int, str]] = [
            (0, 5, "relu1_1"),
            (5, 10, "relu2_1"),
            (10, 19, "relu3_1"),
            (19, 28, "relu4_1"),
        ]
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def forward_multi(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        h = x
        for start, end, _name in self.slice_ids:
            for idx in range(start, end):
                h = self.net[idx](h)
            feats.append(h)
        return feats


def pil_resize_tensor(pil: Image.Image, size: int, device: torch.device) -> torch.Tensor:
    pil = pil.convert("RGB").resize((size, size), Image.BICUBIC)
    t = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    return t


class GramStyleDistance:
    """Reusable VGG Gram distance (sum of Frobenius norms over style layers)."""

    def __init__(self, device: torch.device, image_size: int = 256):
        self.device = device
        self.image_size = image_size
        self.enc = VGGStyleEncoder(device)

    @torch.no_grad()
    def __call__(self, pil_a: Image.Image, pil_b: Image.Image) -> float:
        xa = self.enc.norm(pil_resize_tensor(pil_a, self.image_size, self.device))
        xb = self.enc.norm(pil_resize_tensor(pil_b, self.image_size, self.device))
        fa = self.enc.forward_multi(xa)
        fb = self.enc.forward_multi(xb)
        dist = 0.0
        for a, b in zip(fa, fb):
            ga = _gram(a)
            gb = _gram(b)
            dist += float((ga - gb).pow(2).sum().sqrt().item())
        return dist
