"""根据识别结果生成一段建筑风格描述。

纯规则模板：不调用任何模型或外部服务，同样的输入永远得到同样的文字，
方便论文复现和批量统计。生成结果会写进 result.txt / result.json / report.html。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Tuple

# 整体风格 -> 形容词
_ADJECTIVES: Dict[str, str] = {
    "Neoclassism": "Neoclassical",
    "Classicalism": "Classical",
}

# 元素标签 -> (单数短语, 复数短语)
_ELEMENT_PHRASES: Dict[str, Tuple[str, str]] = {
    "GreekTemple": ("a Greek-temple front", "Greek-temple fronts"),
    "Tempietto": ("a tempietto-like front", "tempietto-like fronts"),
    "St. Peter's Basilica": ("a St Peter's Basilica-like front", "St Peter's Basilica-like fronts"),
    "RomanPantheon": ("a Roman Pantheon-like front", "Roman Pantheon-like fronts"),
    "colonnade": ("a colonnade", "colonnades"),
    "arcade": ("an arcade", "arcades"),
    "pediment": ("a pediment", "pediments"),
    "arch": ("an arch", "arches"),
    "tower": ("a tower", "towers"),
    "dome": ("a dome", "domes"),
    "roseWindow": ("a rose window", "rose windows"),
    "scroll": ("a volute", "volutes"),
    "single_column": ("a free-standing column", "free-standing columns"),
    "Coupled": ("a coupled column pair", "coupled column pairs"),
    "Triple": ("a triple column group", "triple column groups"),
}

_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _number(value: int) -> str:
    return _NUMBERS.get(value, str(value))


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _style_name(name: str) -> str:
    return _ADJECTIVES.get(name, name)


def _join(parts: Sequence[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'。"""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _elements_clause(stages: Sequence[dict]) -> str:
    """把各阶段的检测结果压缩成 'a colonnade, a pediment (Triangular), and ...'。"""
    counts: Counter = Counter()
    styles: Dict[str, Counter] = {}

    for stage in stages:
        for detection in stage["detections"]:
            label = detection["label"]
            counts[label] += 1
            style = detection.get("style")
            if style:
                styles.setdefault(label, Counter())[style["name"]] += 1

    phrases = []
    for label, count in counts.most_common():
        singular, plural = _ELEMENT_PHRASES.get(
            label, (label, label if label.endswith("s") else label + "s"))
        phrase = singular if count == 1 else f"{_number(count)} {plural}"

        notes = styles.get(label)
        if notes:
            if len(notes) == 1:
                phrase += f" ({notes.most_common(1)[0][0]})"
            else:
                detail = ", ".join(f"{name} ×{n}" for name, n in notes.most_common())
                phrase += f" ({detail})"
        phrases.append(phrase)

    return _join(phrases)


def _implicit_clause(stages: Sequence[dict]) -> str:
    """元素风格隐含的其它时期风格。"""
    hints: Counter = Counter()
    for stage in stages:
        for detection in stage["detections"]:
            style = detection.get("style")
            if style and style.get("implicit"):
                hints[(style["implicit"], style["name"])] += 1

    if not hints:
        return ""
    parts = [f"{name} ({implicit} influence) ×{count}"
             for (implicit, name), count in hints.most_common()]
    return "Some elements also point to other styles: " + ", ".join(parts) + "."


def describe(style: dict, stages: Sequence[dict]) -> str:
    """生成一段 2–4 句的建筑风格描述。"""
    top = _style_name(style["top"])
    confidence = _percent(style["prob"])
    sentences = [f"The building is {top} (confidence {confidence})."]

    if not style.get("is_neoclassism"):
        sentences.append(
            "Element-level detection was skipped because the style is not Neoclassical."
        )
        return " ".join(sentences)

    elements = _elements_clause(stages)
    if elements:
        sentences.append(f"The façade presents {elements}.")
    else:
        sentences.append("No architectural elements were detected above the threshold.")

    implicit = _implicit_clause(stages)
    if implicit:
        sentences.append(implicit)

    if style["prob"] < 0.6:
        sentences.append(
            f"The holistic classification is not decisive ({confidence}), "
            "so treat the element list as indicative."
        )

    return " ".join(sentences)
