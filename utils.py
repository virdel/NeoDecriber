"""推理工具：模型缓存、整体风格分类、元素检测与元素风格细分。

本模块只做计算与图片产出（标注图、裁剪图），把结果组织成字典返回；
result.txt / result.json 的排版由 main.py 负责。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from draw_box_utils import draw_box, get_element_images
from model import create_model, resnet34

_CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_DETECT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


class TaskConfig(NamedTuple):
    """一个检测任务的输出配置。"""

    task_string: str     # 标注图文件名（不含扩展名）
    image_dir: str       # 裁剪图子目录
    thresh: float        # 绘制与裁剪的置信度阈值


TASK_CONFIGS: Dict[int, TaskConfig] = {
    1: TaskConfig("imitation element combo", "element_combination", 0.5),
    2: TaskConfig("key_element", "key_element", 0.5),
    3: TaskConfig("arch_element", "arch_element", 0.5),
    4: TaskConfig("column_combination", "column_combination", 0.6),
    5: TaskConfig("single_column", "single_column", 0.5),
}

# 元素标签 -> 子分类器权重目录（与检测权重目录同级）
CLASSIFIER_DIRS: Dict[str, str] = {
    "pediment": "pediment",
    "tower": "tower",
    "dome": "dome",
    "arch": "Arch_classification",
    "single_column": "column_order",
}

# 子分类器的隐含风格。权重里的类名大小写并不统一（如 "Multilayer" / "multilayer"），
# 因此统一转小写后匹配。
_IMPLICIT_STYLES: Dict[str, str] = {
    "broken/semicircular": "Baroque",
    "polygonal": "Baroque",
    "pointed": "Gothic",
    "multilayer": "Romanesque",
}


def implicit_style(class_name: str) -> Optional[str]:
    """元素风格隐含的整体风格；没有对应关系时返回 None。"""
    return _IMPLICIT_STYLES.get(class_name.lower())


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_file(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what}不存在: {path}")
    return path


def _to_tensor(image: Image.Image, transform) -> torch.Tensor:
    return torch.unsqueeze(transform(image.convert("RGB")), dim=0)


def _softmax_predict(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    return torch.softmax(logits, dim=1).cpu()


def _style_predictions(probs: torch.Tensor, class_dict: dict) -> List[dict]:
    """把一批 softmax 概率转成 [{'name', 'prob', 'implicit'}]。"""
    predictions = []
    for row in probs:
        index = int(torch.argmax(row).item())
        name = class_dict[str(index)]
        predictions.append({
            "name": name,
            "prob": round(float(row[index].item()), 6),
            "implicit": implicit_style(name),
        })
    return predictions


# --------------------------------------------------------------------------
# 模型缓存
# --------------------------------------------------------------------------
class ModelCache:
    """按 (权重目录, 设备) 复用已加载的模型，并限制常驻的检测模型数量。

    检测模型约百兆级别，全部常驻会明显推高内存，因此按最近最少使用淘汰。
    """

    def __init__(self, max_detectors: int = 2, max_classifiers: int = 8):
        self.max_detectors = max_detectors
        self.max_classifiers = max_classifiers
        self.classifier_loads = 0
        self.detector_loads = 0
        self._classifiers: "OrderedDict[tuple, Tuple[torch.nn.Module, dict]]" = OrderedDict()
        self._detectors: "OrderedDict[tuple, Tuple[torch.nn.Module, dict]]" = OrderedDict()

    @staticmethod
    def _key(data_root: str, device: torch.device) -> tuple:
        return os.path.normcase(os.path.abspath(data_root)), str(device)

    def classifier(self, data_root: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
        key = self._key(data_root, device)
        cached = self._classifiers.get(key)
        if cached is not None:
            self._classifiers.move_to_end(key)
            return cached

        class_dict = _read_json(_require_file(
            os.path.join(data_root, "class_indices.json"), "类别文件"))
        model = resnet34(num_classes=len(class_dict))
        weights = _require_file(os.path.join(data_root, "resNet34.pth"), "分类模型权重")
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device).eval()

        self.classifier_loads += 1
        self._classifiers[key] = (model, class_dict)
        while len(self._classifiers) > self.max_classifiers:
            self._classifiers.popitem(last=False)
        return model, class_dict

    def detector(self, data_root: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
        key = self._key(data_root, device)
        cached = self._detectors.get(key)
        if cached is not None:
            self._detectors.move_to_end(key)
            return cached

        class_dict = _read_json(_require_file(
            os.path.join(data_root, "my_source_data_class2num.json"), "类别文件"))
        model = create_model(num_classes=len(class_dict) + 1)
        weights = _require_file(os.path.join(data_root, "resNetFpn-model.pth"), "检测模型权重")
        model.load_state_dict(torch.load(weights, map_location=device)["model"])
        model.to(device).eval()

        self.detector_loads += 1
        self._detectors[key] = (model, class_dict)
        while len(self._detectors) > self.max_detectors:
            self._detectors.popitem(last=False)
        return model, class_dict

    def clear(self) -> None:
        self._classifiers.clear()
        self._detectors.clear()
        self.classifier_loads = 0
        self.detector_loads = 0


_CACHE = ModelCache()


def clear_model_cache() -> None:
    """清空模型缓存（测试或显存/内存吃紧时使用）。"""
    _CACHE.clear()


# --------------------------------------------------------------------------
# 整体风格分类
# --------------------------------------------------------------------------
def predict_style(img_path: str, device: torch.device, data_root: str) -> dict:
    """整体风格分类，返回 {'top', 'prob', 'classes', 'is_neoclassism'}。"""
    model, class_dict = _CACHE.classifier(data_root, device)

    img = _to_tensor(Image.open(img_path), _CLASSIFY_TRANSFORM)
    probs = _softmax_predict(model, img.to(device))[0]

    classes = [{"name": class_dict[str(i)], "prob": round(float(probs[i].item()), 6)}
               for i in range(len(probs))]
    best = max(classes, key=lambda item: item["prob"])
    return {
        "top": best["name"],
        "prob": best["prob"],
        "classes": classes,
        "is_neoclassism": best["name"] == "Neoclassism",
    }


def classify_crops(images: Sequence[Image.Image], device: torch.device,
                   data_root: str) -> List[dict]:
    """批量分类多张裁剪图，返回与输入顺序一致的风格列表。

    一次前向即可处理同类的所有裁剪图，避免逐张推理的开销。
    """
    if not images:
        return []
    model, class_dict = _CACHE.classifier(data_root, device)
    batch = torch.cat([_to_tensor(img, _CLASSIFY_TRANSFORM) for img in images], dim=0)
    return _style_predictions(_softmax_predict(model, batch.to(device)), class_dict)


# --------------------------------------------------------------------------
# 元素检测
# --------------------------------------------------------------------------
def _split_pediment(predictions: dict, category_index: dict):
    """把 pediment 单独拆出来，只保留面积最大的一个，其余按原顺序返回。"""
    boxes = predictions["boxes"].to("cpu").numpy().tolist()
    classes = predictions["labels"].to("cpu").numpy().tolist()
    scores = predictions["scores"].to("cpu").numpy().tolist()

    kept_boxes, kept_classes, kept_scores = [], [], []
    best_pediment = None
    best_area = -1.0

    for box, label, score in zip(boxes, classes, scores):
        if category_index[label] == "pediment":
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best_area, best_pediment = area, (box, label, score)
        else:
            kept_boxes.append(box)
            kept_classes.append(label)
            kept_scores.append(score)

    if best_pediment is not None:
        kept_boxes.append(best_pediment[0])
        kept_classes.append(best_pediment[1])
        kept_scores.append(best_pediment[2])

    return (np.array(kept_boxes),
            np.array(kept_classes, dtype=np.int64),
            np.array(kept_scores))


def detect_elements(img_path: str, device: torch.device, save_dir: str, data_root: str,
                    task: int, line_thickness: int, font_size: int) -> dict:
    """执行一个阶段的元素检测，返回该阶段的结构化结果。"""
    config = TASK_CONFIGS[task]
    model, class_dict = _CACHE.detector(data_root, device)
    category_index = {v: k for k, v in class_dict.items()}

    original_img = Image.open(img_path).convert("RGB")
    img = _to_tensor(original_img, _DETECT_TRANSFORM)

    with torch.no_grad():
        predictions = model(img.to(device))[0]

    boxes, classes, scores = _split_pediment(predictions, category_index)
    items = get_element_images(original_img, boxes, classes, scores,
                               category_index, thresh=config.thresh)

    stage = {
        "task": task,
        "threshold": config.thresh,
        "annotated_image": None,
        "detections": [],
    }
    if not items[0]:
        return stage

    draw_box(original_img, boxes, classes, scores, category_index,
             thresh=config.thresh,
             line_thickness=line_thickness,
             font_size=font_size)
    original_img.save(os.path.join(save_dir, config.task_string + ".jpg"))
    stage["annotated_image"] = config.task_string + ".jpg"

    crop_dir = os.path.join(save_dir, config.image_dir)
    os.makedirs(crop_dir, exist_ok=True)

    crop_imgs, names, labels, item_scores, item_boxes = items
    detections = []
    for crop, name, label, score, box in zip(crop_imgs, names, labels, item_scores, item_boxes):
        crop_file = name + ".jpg"
        crop.save(os.path.join(crop_dir, crop_file))
        detections.append({
            "label": label,
            "score": round(float(score), 4),
            "bbox": [round(float(value), 1) for value in box],
            "crop": os.path.relpath(os.path.join(crop_dir, crop_file), save_dir).replace(os.sep, "/"),
            "style": None,
        })

    # 同一子分类器的裁剪图合并成一次前向
    groups: "OrderedDict[Optional[str], List[int]]" = OrderedDict()
    for index, detection in enumerate(detections):
        groups.setdefault(CLASSIFIER_DIRS.get(detection["label"]), []).append(index)

    for classifier_dir, indices in groups.items():
        if classifier_dir is None:
            continue
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(data_root)), classifier_dir)
        try:
            styles = classify_crops([crop_imgs[i] for i in indices], device, weights_dir)
        except FileNotFoundError as exc:
            print(f"[warn] 跳过 {classifier_dir} 风格细分: {exc}")
            continue
        for index, style in zip(indices, styles):
            detections[index]["style"] = style

    stage["detections"] = detections
    return stage


# --------------------------------------------------------------------------
# 计算指标（保留原有工具函数）
# --------------------------------------------------------------------------
def IoU(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoU ratio (intersection over union) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersection
    return float(intersection) / union if union else 0.0


def IoM(box1, box2):
    """
    :param box1: list in format [xmin1, ymin1, xmax1, ymax1]
    :param box2:  list in format [xmin2, ymin2, xamx2, ymax2]
    :return:    returns IoM ratio (intersection over min-area) of two boxes
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    intersection = x_overlap * y_overlap
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    min_area = min(box1_area, box2_area)
    return float(intersection) / min_area if min_area else 0.0
