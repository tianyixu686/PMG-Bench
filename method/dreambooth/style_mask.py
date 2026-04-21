import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class StyleMaskTerms:
    # phrases/regex here represent *style/appearance* tokens to be masked.
    phrases: Tuple[str, ...]
    regex: Tuple[str, ...]


_DEFAULT_TERMS = StyleMaskTerms(
    phrases=(
        # English (style / brushwork / lighting / rendering / texture)
        "in the style of",
        "style of",
        "oil painting",
        "watercolor",
        "gouache",
        "ink wash",
        "pencil sketch",
        "charcoal sketch",
        "line art",
        "anime style",
        "comic style",
        "pixar style",
        "3d render",
        "octane render",
        "unreal engine",
        "cinematic lighting",
        "dramatic lighting",
        "studio lighting",
        "soft lighting",
        "hard lighting",
        "rim light",
        "backlight",
        "volumetric lighting",
        "global illumination",
        "high contrast",
        "low contrast",
        "high saturation",
        "low saturation",
        "film grain",
        "depth of field",
        "bokeh",
        "hdr",
        "photorealistic",
        "hyperrealistic",
        "brush strokes",
        "brushstroke",
        "texture",
        "textured",
        "material",
        # Chinese
        "风格",
        "画风",
        "油画",
        "水彩",
        "水粉",
        "国画",
        "素描",
        "铅笔",
        "线稿",
        "漫画风",
        "动漫风",
        "卡通风",
        "3D渲染",
        "渲染",
        "电影感",
        "光影",
        "柔光",
        "硬光",
        "逆光",
        "轮廓光",
        "体积光",
        "高对比",
        "低对比",
        "高饱和",
        "低饱和",
        "胶片颗粒",
        "景深",
        "散景",
        "笔触",
        "肌理",
        "质感",
        "材质",
        "纹理",
    ),
    regex=(
        # common prompt boilerplate that often leaks style cues
        r"\b(masterpiece|best\s+quality|ultra\s+high\s+res|8k|4k)\b",
        r"\b(highly\s+detailed|ultra\s+detailed)\b",
        r"\b(artstation|deviantart|trending)\b",
        # "in the style of X" patterns
        r"\bin\s+the\s+style\s+of\s+[^,.;]+",
    ),
)


def load_style_mask_terms(terms_path: Optional[str]) -> StyleMaskTerms:
    if not terms_path:
        return _DEFAULT_TERMS

    p = Path(terms_path)
    if not p.exists():
        return _DEFAULT_TERMS

    obj = json.loads(p.read_text(encoding="utf-8"))
    phrases = tuple(str(x) for x in (obj.get("phrases") or []) if str(x).strip())
    regex = tuple(str(x) for x in (obj.get("regex") or []) if str(x).strip())
    if not phrases and not regex:
        return _DEFAULT_TERMS
    return StyleMaskTerms(phrases=phrases or _DEFAULT_TERMS.phrases, regex=regex or _DEFAULT_TERMS.regex)


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\t\r\n]+", " ", str(text))
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    return text.strip(" \t\n\r,.;，。；:：")


def _token_regex(token: str) -> re.Pattern:
    # CLIP tokenizer is BPE, but we still keep a strict "word-like" boundary for safety.
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")


def _ensure_token_exactly_once(text: str, token: str) -> str:
    pat = _token_regex(token)
    matches = list(pat.finditer(text))
    if len(matches) <= 1:
        return text

    # Remove all but first occurrence
    first = matches[0]
    out_parts = []
    last = 0
    kept = False
    for m in matches:
        out_parts.append(text[last : m.start()])
        if not kept:
            out_parts.append(token)
            kept = True
        # else: drop this occurrence
        last = m.end()
    out_parts.append(text[last:])
    out = "".join(out_parts)
    return _normalize(out)


def mask_prompt_to_instance_token(
    prompt: str,
    *,
    instance_token: str = "sks",
    terms_path: Optional[str] = None,
    enabled: bool = True,
    append_if_missing: bool = True,
) -> str:
    """Mask style/appearance tokens by replacing them with a single instance_token.

    Constraints (DreamBooth-style):
    - The output must contain instance_token.
    - The output must contain instance_token **exactly once**.
    - If no style/appearance term is detected, append: "in {token} style".
    """

    original = _normalize(str(prompt or ""))
    if not enabled:
        return original

    token = str(instance_token).strip() or "sks"
    terms = load_style_mask_terms(terms_path)

    text = original

    # 1) Find the earliest style/appearance match to replace with token
    best = None  # (start, end, length)

    def consider(start: int, end: int):
        nonlocal best
        if start < 0 or end <= start:
            return
        length = end - start
        if best is None:
            best = (start, end, length)
            return
        b0, b1, bl = best
        if start < b0 or (start == b0 and length > bl):
            best = (start, end, length)

    # phrase matches
    for ph in terms.phrases:
        ph = str(ph).strip()
        if not ph:
            continue
        flags = re.IGNORECASE if re.search(r"[A-Za-z0-9]", ph) else 0
        for m in re.finditer(re.escape(ph), text, flags=flags):
            consider(m.start(), m.end())
            break

    # regex matches
    for rgx in terms.regex:
        try:
            m = re.search(rgx, text, flags=re.IGNORECASE)
        except re.error:
            continue
        if m:
            consider(m.start(), m.end())

    if best is not None:
        s, e, _ = best
        text = (text[:s] + f" {token} " + text[e:]).strip()

        # 2) Remove remaining style/appearance terms to avoid leaking
        # (do this AFTER inserting token so we don't delete the append template)
        for ph in sorted((p for p in terms.phrases if str(p).strip()), key=lambda x: len(str(x)), reverse=True):
            ph = str(ph).strip()
            flags = re.IGNORECASE if re.search(r"[A-Za-z0-9]", ph) else 0
            text = re.sub(re.escape(ph), " ", text, flags=flags)
        for rgx in terms.regex:
            try:
                text = re.sub(rgx, " ", text, flags=re.IGNORECASE)
            except re.error:
                continue

        text = _normalize(text)

    # 3) Ensure token exists exactly once (append if missing)
    pat = _token_regex(token)
    if not pat.search(text):
        if append_if_missing:
            text = _normalize(f"{text} in {token} style") if text else f"in {token} style"
        else:
            # Hard guarantee: still inject token
            text = _normalize(f"{text} {token}") if text else token

    text = _ensure_token_exactly_once(text, token)
    return text


# Backward-compatible name (previously used by wrappers)
def mask_style_prompt(prompt: str, *, terms_path: Optional[str] = None, enabled: bool = True) -> str:
    return mask_prompt_to_instance_token(prompt, instance_token="sks", terms_path=terms_path, enabled=enabled)
