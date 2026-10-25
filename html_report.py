"""把 result.json 的数据渲染成一个自包含的 HTML 报告。

不依赖 Web 服务器、不引用任何 CDN：CSS/JS 全部内联，图片用相对路径，
双击 report.html 就能离线查看。数据量较大时（几百个裁剪图）仍是静态页面，
浏览器懒加载图片即可。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

REPORT_NAME = "report.html"

_CSS = """
:root {
  --bg: #f6f4f0; --card: #fff; --ink: #1d1c1a; --muted: #6f6a62;
  --line: #e3ded5; --accent: #9a5b33; --mark: #c2410c;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.6 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
a { color: inherit; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 72px; }
.top {
  display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline;
  justify-content: space-between; border-bottom: 1px solid var(--line); padding-bottom: 16px;
}
.brand { font-family: Georgia, "Times New Roman", serif; font-size: 22px; letter-spacing: .01em; }
.meta { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px 18px; }
.meta b { color: var(--ink); font-weight: 600; }
.hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(300px, .6fr);
        gap: 24px; margin: 28px 0 12px; align-items: start; }
.hero figure { margin: 0; }
.hero img, .annotated img {
  width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line);
  background: #fff;
}
figcaption { color: var(--muted); font-size: 12px; margin-top: 6px; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 20px 22px; }
.panel h3 { margin: 0 0 14px; font: 600 13px/1.3 inherit; letter-spacing: .08em;
            text-transform: uppercase; color: var(--muted); }
.style-big { font-family: Georgia, "Times New Roman", serif; font-size: 30px; line-height: 1.2; }
.style-prob { color: var(--accent); font-size: 20px; margin-left: 8px; }
.bars { margin: 14px 0 18px; }
.bar-row { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 13px; margin-bottom: 8px; }
.bar-track { grid-column: 1 / -1; height: 5px; background: #efeae2; border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; }
.stats { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.pill { background: #f2ede6; border-radius: 999px; padding: 3px 10px; font-size: 12.5px; }
.pill small { color: var(--muted); }
.notice { background: #fdf3e7; border: 1px solid #f0d9bf; color: #7c4a17;
          border-radius: 6px; padding: 12px 16px; margin: 18px 0 0; font-size: 14px; }
.stage { margin-top: 44px; border-top: 1px solid var(--line); padding-top: 24px; }
.stage h2 { margin: 0; font: 600 19px/1.35 Georgia, "Times New Roman", serif; }
.badge { display: inline-block; margin-left: 10px; font: 500 12px/1 inherit;
         color: var(--muted); background: #f1ece4; border-radius: 999px; padding: 5px 10px; vertical-align: 2px; }
.stage-body { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
              gap: 22px; margin-top: 16px; align-items: start; }
.annotated { position: relative; }
.overlay { position: absolute; border: 2px solid var(--mark); background: rgba(194, 65, 12, .12);
           display: none; pointer-events: none; border-radius: 2px; }
.overlay.on { display: block; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip { border: 1px solid var(--line); background: var(--card); color: var(--muted);
        border-radius: 999px; padding: 4px 11px; font-size: 12.5px; cursor: pointer; }
.chip.on { background: var(--ink); border-color: var(--ink); color: #fff; }
.crops { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px; }
.crop { margin: 0; border: 1px solid var(--line); border-radius: 6px; overflow: hidden;
        background: var(--card); cursor: pointer; }
.crop img { width: 100%; height: 92px; object-fit: cover; display: block; background: #f4f1ec; }
.crop figcaption { margin: 0; padding: 6px 8px; font-size: 11.5px; line-height: 1.35; color: var(--muted); }
.crop figcaption b { color: var(--ink); font-weight: 600; }
.crop.selected { outline: 2px solid var(--accent); outline-offset: -2px; }
.empty { color: var(--muted); font-size: 14px; margin: 10px 0 0; }
footer { margin-top: 56px; border-top: 1px solid var(--line); padding-top: 16px;
         color: var(--muted); font-size: 12.5px; }
footer code { background: #f1ece4; padding: 1px 5px; border-radius: 4px; }
@media (max-width: 860px) {
  .hero, .stage-body { grid-template-columns: 1fr; }
}
"""

_JS = """
(function () {
  function percent(value, total) { return (value / (total || 1)) * 100 + '%'; }
  document.querySelectorAll('.stage').forEach(function (stage) {
    var img = stage.querySelector('.annotated img');
    var overlay = stage.querySelector('.overlay');
    var cards = Array.prototype.slice.call(stage.querySelectorAll('.crop'));
    if (!img || !overlay || !cards.length) return;

    stage.querySelectorAll('.chip').forEach(function (chip) {
      chip.addEventListener('click', function () {
        stage.querySelectorAll('.chip').forEach(function (c) { c.classList.remove('on'); });
        chip.classList.add('on');
        var label = chip.dataset.label || '';
        cards.forEach(function (card) {
          card.style.display = (!label || card.dataset.label === label) ? '' : 'none';
        });
      });
    });

    cards.forEach(function (card) {
      card.addEventListener('click', function () {
        var box = card.dataset.bbox.split(',').map(Number);
        var w = img.naturalWidth || 1, h = img.naturalHeight || 1;
        overlay.style.left = percent(box[0], w);
        overlay.style.top = percent(box[1], h);
        overlay.style.width = percent(box[2] - box[0], w);
        overlay.style.height = percent(box[3] - box[1], h);
        overlay.classList.add('on');
        cards.forEach(function (c) { c.classList.remove('selected'); });
        card.classList.add('selected');
      });
    });
  });
})();
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _src(path: str) -> str:
    """相对路径转 URL（空格、中文都编码，斜杠保留）。"""
    return quote(str(path).replace("\\", "/"), safe="/")


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _style_bars(classes: Iterable[dict]) -> str:
    rows = []
    for item in classes:
        width = max(float(item["prob"]) * 100, 0.4)
        rows.append(
            '<div class="bar-row">'
            f'<span>{_esc(item["name"])}</span><span>{_pct(item["prob"])}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
            "</div>"
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def _summary_pills(summary: dict) -> str:
    pills = [f'<span class="pill">{int(summary.get("total_detections", 0))} elements</span>']
    for label, count in sorted(summary.get("counts", {}).items(), key=lambda kv: -kv[1]):
        pills.append(f'<span class="pill">{_esc(label)} <small>×{int(count)}</small></span>')
    for item in summary.get("implicit_styles", []):
        pills.append(
            f'<span class="pill">{_esc(item["implicit"])} '
            f'<small>{_esc(item["class"])} ×{int(item["count"])}</small></span>'
        )
    return '<div class="stats">' + "".join(pills) + "</div>"


def _hero(payload: dict) -> str:
    style = payload["style"]
    summary = payload.get("summary", {})
    image = payload.get("input_image")
    figure = ""
    if image:
        figure = (
            '<figure><img src="' + _src(image) + '" alt="input image">'
            '<figcaption>' + _esc(payload["image"]) + "</figcaption></figure>"
        )
    return (
        '<section class="hero">'
        f"{figure}"
        '<div class="panel">'
        "<h3>Holistic style</h3>"
        f'<div class="style-big">{_esc(style["top"])}'
        f'<span class="style-prob">{_pct(style["prob"])}</span></div>'
        f'{_style_bars(style["classes"])}'
        "<h3>Summary</h3>"
        f"{_summary_pills(summary)}"
        "</div></section>"
    )


def _crop_cards(stage: dict) -> str:
    cards = []
    for index, detection in enumerate(stage["detections"]):
        style = detection.get("style")
        style_text = ""
        if style:
            style_text = f'{_esc(style["name"])} {_pct(style["prob"])}'
            if style.get("implicit"):
                style_text += f' · {_esc(style["implicit"])}'
        box = ",".join(f"{float(value):.1f}" for value in detection["bbox"])
        cards.append(
            '<figure class="crop"'
            f' data-label="{_esc(detection["label"])}" data-bbox="{box}"'
            f' title="{_esc(detection["label"])} #{index} · {_pct(detection["score"])}">'
            f'<img src="{_src(detection["crop"])}" alt="{_esc(detection["label"])}" loading="lazy">'
            "<figcaption>"
            f'<b>{_esc(detection["label"])}</b> {_pct(detection["score"])}'
            + (f"<br>{style_text}" if style_text else "")
            + "</figcaption></figure>"
        )
    return '<div class="crops">' + "".join(cards) + "</div>"


def _chips(stage: dict) -> str:
    counts: dict = {}
    for detection in stage["detections"]:
        counts[detection["label"]] = counts.get(detection["label"], 0) + 1
    chips = [f'<button class="chip on" data-label="">All ({len(stage["detections"])})</button>']
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        chips.append(
            f'<button class="chip" data-label="{_esc(label)}">{_esc(label)} ({count})</button>'
        )
    return '<div class="chips">' + "".join(chips) + "</div>"


def _stage(number: int, stage: dict) -> str:
    detections = stage["detections"]
    annotated = stage.get("annotated_image")
    if detections:
        badge = f'<span class="badge">{len(detections)} items</span>'
    else:
        badge = '<span class="badge">no object detected</span>'

    heading = f'<h2>[{number}] {_esc(stage["title"])}{badge}</h2>'
    if not detections:
        return f'<section class="stage">{heading}<p class="empty">No element above the threshold.</p></section>'

    figure = ""
    if annotated:
        thumb = stage.get("annotated_thumb") or annotated
        figure = (
            '<figure class="annotated">'
            f'<a href="{_src(annotated)}" target="_blank" rel="noopener">'
            f'<img src="{_src(thumb)}" alt="{_esc(stage["title"])}" loading="lazy"></a>'
            '<div class="overlay"></div>'
            "<figcaption>Click an element to highlight its box; click the image to open it full size.</figcaption>"
            "</figure>"
        )

    return (
        f'<section class="stage">{heading}'
        f'<div class="stage-body">{figure}'
        f"<div>{_chips(stage)}{_crop_cards(stage)}</div>"
        "</div></section>"
    )


def render(payload: dict) -> str:
    """根据 result payload 生成完整的 HTML 文本。"""
    title = "NeoDecriber result"
    meta = (
        f'<span>device <b>{_esc(payload.get("device", "-"))}</b></span>'
        f'<span>elapsed <b>{float(payload.get("elapsed_seconds", 0)):.2f} s</b></span>'
        f'<span>created <b>{_esc(payload.get("created_at", "-"))}</b></span>'
    )

    stages = "".join(
        _stage(number, stage) for number, stage in enumerate(payload.get("stages", []), start=2)
    )
    notice = ""
    if not payload.get("style", {}).get("is_neoclassism", False):
        notice = (
            '<p class="notice">The holistic style is not <b>Neoclassism</b>, '
            "so element detection was skipped.</p>"
        )

    footer = (
        f"<footer>Output <code>{_esc(payload.get('output_dir', '-'))}</code> &nbsp;·&nbsp; "
        f"weights <code>{_esc(payload.get('source_dir', '-'))}</code> &nbsp;·&nbsp; "
        f"data <a href=\"result.json\">result.json</a> &nbsp;·&nbsp; "
        f"report <a href=\"result.txt\">result.txt</a></footer>"
    )

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        '<div class="wrap">'
        f'<header class="top"><div class="brand">{title}</div><div class="meta">{meta}</div></header>'
        f"{_hero(payload)}"
        f"{notice}"
        f"{stages}"
        f"{footer}"
        "</div>\n"
        f"<script>{_JS}</script>\n"
        "</body>\n</html>\n"
    )


def write(output_dir: Path, payload: dict) -> Path:
    """写出 report.html 并返回路径。"""
    target = Path(output_dir) / REPORT_NAME
    target.write_text(render(payload), encoding="utf-8")
    return target
