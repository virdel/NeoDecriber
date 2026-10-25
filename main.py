"""NeoDecriber 命令行入口：识别并描述新古典主义建筑。

示例:
    python main.py --img-path ./Dublin.jpg --output-dir ./result

流程（与论文的级联模型一致）:
    1. 整体风格分类 (ResNet34)
    2. 仿古元素组合检测
    3. 关键元素检测
    4. 拱券元素检测
    5. 柱式组合检测
    6. 单柱检测

输出:
    <output-dir>/result.txt    人类可读报告（分节 + 表格 + 汇总）
    <output-dir>/result.json   同一批数据的结构化版本，便于批量统计
    <output-dir>/*.jpg         各阶段标注图
    <output-dir>/<stage>/*.jpg 裁剪出的元素图

所有相对路径的权重目录都相对本文件解析，因此可以在任意工作目录下运行。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import html_report

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "source"
RESULT_TXT = "result.txt"
RESULT_JSON = "result.json"
RULE = "—" * 42
STAR = "*" * 16


class Stage(NamedTuple):
    """一个元素检测阶段：任务号、权重子目录、控制台/报告标题。"""

    task_id: int
    weight_dir: str
    title: str


STAGES = (
    Stage(1, "imitation", "Imitation element combo detection"),
    Stage(2, "key_element", "Key element detection"),
    Stage(3, "arch", "Arch element detection"),
    Stage(4, "column_combination", "Column combination detection"),
    Stage(5, "single_column", "Single column detection"),
)

# 仅在检测出对应标签时才会用到的子分类器，缺失只提示不拦截
OPTIONAL_CLASSIFIERS = ("pediment", "tower", "dome", "Arch_classification", "column_order")


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--img-path",
        required=True,
        metavar="PATH",
        help="待检测图片路径（必填）",
    )
    parser.add_argument(
        "--output-dir",
        default="./result",
        metavar="DIR",
        help="结果保存目录，默认 ./result",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="推理设备，如 cuda:0 / cpu；CUDA 不可用时自动回退到 CPU（默认 cuda:0）",
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        metavar="DIR",
        help="模型权重所在目录，默认 <项目目录>/source",
    )
    parser.add_argument(
        "--font_size", "--font-size",
        dest="font_size",
        type=_positive_int,
        default=10,
        metavar="N",
        help="标注框文字字号，默认 10",
    )
    parser.add_argument(
        "--line_thickness", "--line-thickness",
        dest="line_thickness",
        type=_positive_int,
        default=4,
        metavar="N",
        help="标注框线宽，默认 4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录已存在时先清空；不加该参数则保留其中已有文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只做参数、路径与权重检查并打印执行计划，不加载模型",
    )
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"应为正整数，收到 {text!r}")
    return value


# --------------------------------------------------------------------------
# 检查
# --------------------------------------------------------------------------
def check_image(img_path: Path) -> None:
    if not img_path.is_file():
        raise SystemExit(f"图片不存在: {img_path}")


def check_weights(source_dir: Path) -> None:
    """提前校验权重文件，避免跑到一半才在 assert 里失败。"""
    required: List[Path] = [
        source_dir / "holistic_style" / "class_indices.json",
        source_dir / "holistic_style" / "resNet34.pth",
    ]
    for stage in STAGES:
        required.append(source_dir / stage.weight_dir / "my_source_data_class2num.json")
        required.append(source_dir / stage.weight_dir / "resNetFpn-model.pth")

    missing = [path for path in required if not path.is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            f"缺少 {len(missing)} 个模型权重文件，请按 README 下载权重并解压到:\n"
            f"  {source_dir}\n{listing}"
        )

    absent = [
        name
        for name in OPTIONAL_CLASSIFIERS
        if not (source_dir / name / "resNet34.pth").is_file()
    ]
    if absent:
        print(f"[warn] 以下子分类器权重缺失，检测到相应元素时会跳过风格细分: {', '.join(absent)}")


def _is_safe_delete_target(path: Path) -> bool:
    """拒绝把磁盘根目录、用户目录或项目目录（及其上级）当成可清空的目标。"""
    resolved = path.resolve()
    home = Path.home().resolve()
    if resolved == Path(resolved.anchor):
        return False
    if resolved == home or resolved in home.parents:
        return False
    if resolved == REPO_ROOT or resolved in REPO_ROOT.parents:
        return False
    return True


class OutputPlan(NamedTuple):
    """输出目录的处理方式：create 新建 / keep 保留 / clean 先清空。"""

    action: str
    text: str


def plan_output_dir(output_dir: Path, overwrite: bool) -> OutputPlan:
    """判断输出目录会遇到什么情况；目标不安全时直接报错。"""
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"输出路径已存在且不是目录: {output_dir}")
    if not output_dir.is_dir():
        return OutputPlan("create", "将新建")
    if not overwrite:
        return OutputPlan("keep", "已存在，保留已有文件；加 --overwrite 可先清空")
    if not _is_safe_delete_target(output_dir):
        raise SystemExit(f"拒绝清空该目录（可能误删重要文件）: {output_dir}")
    return OutputPlan("clean", "已存在，将先清空")


def prepare_output_dir(output_dir: Path, plan: OutputPlan) -> Path:
    if plan.action == "clean":
        shutil.rmtree(output_dir)
    elif plan.action == "keep":
        print(f"[info] 输出目录已存在，保留已有文件: {output_dir}（加 --overwrite 可先清空）")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_input_image(img_path: Path, output_dir: Path, max_side: int = 1600) -> Optional[str]:
    """把原图缩放后复制进输出目录，让 HTML 报告脱离原图路径也能打开。

    报告只需要看整体效果，因此限制长边像素，避免几千万像素的原图拖慢页面。
    """
    try:
        from PIL import Image

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            target = output_dir / "input.jpg"
            image.save(target, quality=88)
    except Exception as exc:  # 图片损坏、格式不支持等都不该中断推理
        print(f"[warn] 无法生成报告用原图: {exc}")
        return None
    return target.name


def resolve_device(requested: str, torch):
    """把用户给的设备名解析成可用的 torch.device，必要时回退并提示。"""
    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            print(f"[warn] CUDA 不可用，{requested} 回退到 CPU")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError):
        raise SystemExit(f"无法识别的设备: {requested!r}，示例: cuda:0 / cpu")

    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            print(
                f"[warn] 只有 {torch.cuda.device_count()} 块 GPU，"
                f"{requested} 回退到 cuda:0"
            )
            index = 0
        device = torch.device(f"cuda:{index}")
    return device


def make_report_images(output_dir: Path, stages: Sequence[dict], max_side: int = 1400) -> None:
    """为 HTML 报告生成缩小的标注图，避免整页加载几十兆的原图。"""
    try:
        from PIL import Image
    except ImportError:
        return

    folder = output_dir / "report"
    for stage in stages:
        name = stage.get("annotated_image")
        if not name:
            continue
        source = output_dir / name
        if not source.is_file():
            continue
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                folder.mkdir(exist_ok=True)
                target = folder / Path(name).name
                image.save(target, quality=80)
            stage["annotated_thumb"] = f"report/{Path(name).name}"
        except Exception as exc:  # 缩略图失败不该影响主结果
            print(f"[warn] 无法生成报告缩略图 {name}: {exc}")


# --------------------------------------------------------------------------
# 报告排版
# --------------------------------------------------------------------------
def _percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _counts_text(counter: Counter) -> str:
    return ", ".join(f"{name} {count}" for name, count in counter.most_common())


def _style_text(style: Optional[dict]) -> str:
    if not style:
        return "-"
    return f"{style['name']} {_percent(style['prob'], 1)}"


def _implicit_counter(stages: Sequence[dict]) -> Counter:
    counter: Counter = Counter()
    for stage in stages:
        for detection in stage["detections"]:
            style = detection.get("style")
            if style and style.get("implicit"):
                counter[(style["implicit"], style["name"])] += 1
    return counter


def _implicit_text(counter: Counter) -> str:
    return ", ".join(
        f"{implicit} {count} ({name})" for (implicit, name), count in counter.most_common()
    )


def build_report(meta: dict, style: dict, stages: Sequence[dict]) -> List[str]:
    """生成 result.txt 的行列表。"""
    lines = [
        "NeoDecriber result",
        "=" * 17,
        f"image   : {meta['image']}",
        f"source  : {meta['source']}",
        f"device  : {meta['device']}",
        f"time    : {meta['time']}",
        "",
        "[1] Holistic style classification",
        f"    {style['top']}  {_percent(style['prob'])}",
        "    probabilities : " + " | ".join(
            f"{item['name']} {_percent(item['prob'])}" for item in style["classes"]),
    ]

    for number, stage in enumerate(stages, start=2):
        detections = stage["detections"]
        if detections:
            counts = Counter(detection["label"] for detection in detections)
            heading = (f"[{number}] {stage['title']}   "
                       f"{len(detections)} items ({_counts_text(counts)})")
        else:
            heading = f"[{number}] {stage['title']}   no object detected"
        lines.append("")
        lines.append(heading)
        if not detections:
            continue

        lines.append(f"    {'#':>3}  {'label':<16}{'score':>7}  {'style':<26}crop")
        for index, detection in enumerate(detections):
            lines.append(
                f"    {index:>3}  {detection['label']:<16}"
                f"{_percent(detection['score'], 1):>7}  "
                f"{_style_text(detection['style']):<26}{detection['crop']}"
            )

    all_detections = [detection for stage in stages for detection in stage["detections"]]
    counts = Counter(detection["label"] for detection in all_detections)
    implicit = _implicit_counter(stages)

    lines.append("")
    lines.append("Summary")
    lines.append(f"    style           : {style['top']} ({_percent(style['prob'])})")
    lines.append(f"    elements        : {_counts_text(counts) if counts else '-'}")
    lines.append(f"    implicit styles : {_implicit_text(implicit) if implicit else '-'}")
    lines.append(f"    elapsed         : {meta['elapsed']:.2f} s")
    return lines


def build_payload(meta: dict, style: dict, stages: Sequence[dict]) -> dict:
    """生成 result.json 的内容。"""
    all_detections = [detection for stage in stages for detection in stage["detections"]]
    counts = Counter(detection["label"] for detection in all_detections)
    implicit = _implicit_counter(stages)

    return {
        "image": meta["image"],
        "output_dir": meta["output_dir"],
        "source_dir": meta["source"],
        "device": meta["device"],
        "created_at": meta["time_iso"],
        "elapsed_seconds": round(meta["elapsed"], 2),
        "input_image": meta.get("input_image"),
        "style": style,
        "stages": list(stages),
        "summary": {
            "total_detections": len(all_detections),
            "counts": dict(counts),
            "implicit_styles": [
                {"implicit": implicit_name, "class": class_name, "count": count}
                for (implicit_name, class_name), count in implicit.most_common()
            ],
        },
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(args: argparse.Namespace) -> int:
    img_path = Path(args.img_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()

    check_image(img_path)
    check_weights(source_dir)
    output_plan = plan_output_dir(output_dir, args.overwrite)

    if args.dry_run:
        print("[dry-run] 检查通过，未加载模型。")
        print(f"  图片    : {img_path}")
        print(f"  输出目录: {output_dir}（{output_plan.text}）")
        print(f"  权重目录: {source_dir}")
        print(f"  设备    : {args.device}")
        print("  阶段    :")
        for stage in STAGES:
            print(f"    {stage.task_id}. {stage.title}")
        return 0

    output_dir = prepare_output_dir(output_dir, output_plan)

    # 延迟导入：--help / --dry-run 不需要 PyTorch，缺失时也能给出明确提示
    try:
        import torch
        import utils
    except ImportError as exc:
        raise SystemExit(f"导入依赖失败（{exc}），请先执行: pip install -r requirements.txt")

    input_image = copy_input_image(img_path, output_dir)

    device = resolve_device(args.device, torch)
    print(f"Using {device.type} device.")
    print(f"图片: {img_path}")
    print(f"输出: {output_dir}")

    started_at = datetime.now()
    started = time.perf_counter()
    timings: List[tuple] = []

    print(RULE)
    print("Holistic style classification:")
    step = time.perf_counter()
    style = utils.predict_style(str(img_path), device, str(source_dir / "holistic_style"))
    timings.append(("Holistic style classification", time.perf_counter() - step))
    print(f"  {style['top']}  {_percent(style['prob'])}")

    stages: List[dict] = []
    if not style["is_neoclassism"]:
        print("风格分类结果不是 Neoclassism，跳过元素检测。")
    else:
        for stage in STAGES:
            print(RULE)
            print(stage.title + ":")
            step = time.perf_counter()
            result = utils.detect_elements(
                str(img_path),
                device,
                str(output_dir),
                str(source_dir / stage.weight_dir),
                stage.task_id,
                args.line_thickness,
                args.font_size,
            )
            timings.append((stage.title, time.perf_counter() - step))
            result["title"] = stage.title
            stages.append(result)
            if result["detections"]:
                counts = Counter(d["label"] for d in result["detections"])
                print(f"  {len(result['detections'])} items ({_counts_text(counts)})")
            else:
                print("  no object detected")

    elapsed = time.perf_counter() - started
    make_report_images(output_dir, stages)
    meta = {
        "image": str(img_path),
        "output_dir": str(output_dir),
        "source": str(source_dir),
        "device": str(device),
        "time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "time_iso": started_at.isoformat(timespec="seconds"),
        "elapsed": elapsed,
        "input_image": input_image,
    }

    txt_path = output_dir / RESULT_TXT
    json_path = output_dir / RESULT_JSON
    payload = build_payload(meta, style, stages)
    txt_path.write_text("\n".join(build_report(meta, style, stages)) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = html_report.write(output_dir, payload)

    print(RULE)
    print(STAR)
    print("Detection complete!")
    print(STAR)
    print(f"结果文件: {txt_path}")
    print(f"          {json_path}")
    print(f"可视化  : {report_path}")
    for title, seconds in timings:
        print(f"  {title:<36}{seconds:6.2f}s")
    print(f"  {'Total':<36}{elapsed:6.2f}s")
    return 0


def cli(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return main(args)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(cli())
