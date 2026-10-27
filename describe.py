"""根据识别结果生成一段建筑风格描述。

输出格式（与论文里的描述方式一致）::

    This is a Neoclassical building. It has imitated Greek temple. It has
    1 triangular pediment, 1 colonnade, and 8 single columns including 5 Ionic
    and 3 Corinthian column orders in the colonnade.

纯规则模板：不调用任何模型或外部服务，同样的输入永远得到同样的文字。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

# 整体风格 -> 形容词
_ADJECTIVES: Dict[str, str] = {
    "Neoclassism": "Neoclassical",
    "Classicalism": "Classical",
}

# 仿古元素（第一阶段）-> 句子里的说法
_IMITATIONS: Dict[str, str] = {
    "GreekTemple": "Greek temple",
    "Tempietto": "Tempietto",
    "St. Peter's Basilica": "St Peter's Basilica",
    "RomanPantheon": "Roman Pantheon",
}

# 元素标签 -> (单数名词, 复数名词)
_ELEMENT_NOUNS: Dict[str, Tuple[str, str]] = {
    "pediment": ("pediment", "pediments"),
    "colonnade": ("colonnade", "colonnades"),
    "arcade": ("arcade", "arcades"),
    "arch": ("arch", "arches"),
    "tower": ("tower", "towers"),
    "dome": ("dome", "domes"),
    "single_column": ("single column", "single columns"),
    "roseWindow": ("rose window", "rose windows"),
    "scroll": ("volute", "volutes"),
    "Coupled": ("coupled column pair", "coupled column pairs"),
    "Triple": ("triple column group", "triple column groups"),
}

# 句子里的出现顺序：自上而下
_ELEMENT_ORDER: Dict[str, int] = {
    "pediment": 10, "colonnade": 20, "arcade": 30, "tower": 40, "dome": 50,
    "arch": 60, "single_column": 70, "Coupled": 80, "Triple": 81,
    "roseWindow": 90, "scroll": 91,
}


def _style_name(name: str) -> str:
    return _ADJECTIVES.get(name, name)


def _join(parts: Sequence[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'；两个元素时不加牛津逗号。"""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] + " and " + parts[1]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _collect(stages: Sequence[dict]):
    """统计元素数量、每个元素的风格分布，以及仿古元素。"""
    counts: Counter = Counter()
    styles: Dict[str, Counter] = {}
    imitations = []
    for stage in stages:
        for detection in stage["detections"]:
            label = detection["label"]
            imitation = _IMITATIONS.get(label)
            if imitation:
                if imitation not in imitations:
                    imitations.append(imitation)
                continue
            counts[label] += 1
            style = detection.get("style")
            if style:
                styles.setdefault(label, Counter())[style["name"]] += 1
    return counts, styles, imitations


def _dominant_adjective(notes: Optional[Counter]) -> str:
    """元素的主风格形容词；"Other" 或多数不占优时返回空串。"""
    if not notes:
        return ""
    name, count = notes.most_common(1)[0]
    if name == "Other" or count * 2 <= sum(notes.values()):
        return ""
    return name.lower()


def _element_phrase(label: str, count: int, notes: Optional[Counter],
                    has_colonnade: bool) -> str:
    singular, plural = _ELEMENT_NOUNS.get(
        label, (label, label if label.endswith("s") else label + "s"))
    noun = singular if count == 1 else plural

    # 柱式的风格属于"柱式"，不能当成单柱的形容词（否则会写出 ionic single columns）
    adjective = "" if label == "single_column" else _dominant_adjective(notes)
    phrase = f"{count} {adjective + ' ' if adjective else ''}{noun}"

    if label == "single_column" and notes:
        orders = _join([f"{n} {name}" for name, n in notes.most_common()])
        phrase += f" including {orders} column orders"
        if has_colonnade:
            phrase += " in the colonnade"
    return phrase


def describe(style: dict, stages: Sequence[dict]) -> str:
    """生成 2–3 句建筑风格描述。"""
    sentences = [f"This is a {_style_name(style['top'])} building."]

    if not style.get("is_neoclassism"):
        sentences.append("Element detection is only run for Neoclassical buildings.")
        return " ".join(sentences)

    counts, styles, imitations = _collect(stages)

    if imitations:
        sentences.append(f"It has imitated {_join(imitations)}.")

    if counts:
        has_colonnade = counts.get("colonnade", 0) > 0
        ordered = sorted(counts.items(),
                         key=lambda kv: (_ELEMENT_ORDER.get(kv[0], 95), -kv[1]))
        phrases = [_element_phrase(label, count, styles.get(label), has_colonnade)
                   for label, count in ordered]
        sentences.append("It has " + _join(phrases) + ".")
    else:
        sentences.append("No architectural elements were detected above the threshold.")

    return " ".join(sentences)
