"""Matplotlib 中文显示与论文图常用文案（专有名词保留英文缩写）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_METHOD_DISPLAY = {
    "ip_adapter": "IP-Adapter",
    "pmg": "PMG",
    "textual_inversion": "Textual Inversion",
    "dreambooth": "DreamBooth",
    "custom_diffusion": "Custom Diffusion",
}

_METRIC_AXIS_ZH = {
    "clip_i": "CLIP-I",
    "clip_t": "CLIP-T",
    "dino_i": "DINO-I",
    "lpips_target": "LPIPS↓",
    "gram_target": "Gram↓",
    "hpsv2": "HPS v2",
}


def method_label(key: str) -> str:
    return _METHOD_DISPLAY.get(key, key)


def metric_axis_label(key: str) -> str:
    return _METRIC_AXIS_ZH.get(key, key)


def setup_matplotlib_chinese() -> None:
    """配置 sans-serif 以渲染中文；若无 CJK 字体则仍设置 unicode_minus。"""
    import matplotlib
    from matplotlib import font_manager as fm

    matplotlib.rcParams["axes.unicode_minus"] = False

    prefer_names: List[str] = []
    for f in fm.fontManager.ttflist:
        n = (f.name or "").lower()
        if any(
            x in n
            for x in (
                "cjk",
                "noto sans cjk",
                "noto serif cjk",
                "simhei",
                "yahei",
                "wenquanyi",
                "zen hei",
                "micro hei",
            )
        ):
            if f.name not in prefer_names:
                prefer_names.append(f.name)

    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(fp).is_file():
            try:
                fm.fontManager.addfont(fp)
                name = fm.FontProperties(fname=fp).get_name()
                if name not in prefer_names:
                    prefer_names.insert(0, name)
            except Exception:
                pass

    if prefer_names:
        matplotlib.rcParams["font.sans-serif"] = prefer_names + ["DejaVu Sans"]
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def cjk_font_properties():
    """返回可用于 text()/suptitle 的 FontProperties；无 CJK 时返回 None。"""
    from matplotlib.font_manager import FontProperties

    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(fp).is_file():
            return FontProperties(fname=fp)
    return None
