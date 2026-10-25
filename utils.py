"""推理工具：模型缓存、整体风格分类、元素检测与元素风格分类。

与论文的级联流程对应：
    整体风格分类 -> 各阶段 Faster R-CNN 元素检测 -> 裁剪图上的 ResNet34 风格细分。

模型按「权重目录 + 设备」缓存复用，避免每张裁剪图都重新构建并加载一次网络。
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

    task_string: str     # 结果图文件名
    image_dir: str       # 结果图子目录
    txt_header: str      # 写入 result.txt 的标题
    thresh: float        # 绘制与裁剪的置信度阈值


TASK_CONFIGS: Dict[int, TaskConfig] = {
    1: TaskConfig("imitation element combo", "element_combination",
                  "imitation element combo:\n", 0.5),
    2: TaskConfig("key_element", "key_element",
                  "key element detection :\n", 0.5),
    3: TaskConfig("arch_element", "arch_element",
                  "arch element detection:\n", 0.5),
    4: TaskConfig("column_combination", "column_combination",
                  "column combination detection:\n", 0.6),
    5: TaskConfig("single_column", "single_column",
                  "single column detection:\n", 0.5),
}

# 元素标签 -> 子分类器权重目录（与检测权重目录同级）
CLASSIFIER_DIRS: Dict[str, str] = {
    "pediment": "pediment",
    "tower": "tower",
    "dome": "dome",
    "arch": "Arch_classification",
    "single_column": "column_order",
}

# 子分类器的隐含风格
IMPLICIT_STYLES: Dict[str, str] = {
    "Broken/semicircular": "Baroque",
    "Polygonal": "Baroque",
    "Pointed": "Gothic",
    "Multilayer": "Romanesque",
}


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
# 分类
# --------------------------------------------------------------------------
def classification(img_path: str, device: torch.device, txt_path: str, data_root: str) -> int:
    """整体风格分类，结果写入 txt_path，返回 1 表示新古典主义。"""
    model, class_dict = _CACHE.classifier(data_root, device)

    img = _to_tensor(Image.open(img_path), _CLASSIFY_TRANSFORM)
    predict = _softmax_predict(model, img.to(device)).squeeze(0)

    predict_cla = int(torch.argmax(predict).item())
    print_res = "class: {}   prob: {:.3}".format(class_dict[str(predict_cla)],
                                                 predict[predict_cla].item())

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_dict[str(i)],
                                                  predict[i].item()))

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("Holistic style classification：\n    ")
        f.write(print_res)
        f.write("\n    Probability of each category：\n")
        for i in range(len(predict)):
            f.write("       class: {:10}   prob: {:.3} \n".format(class_dict[str(i)],
                                                                  predict[i].item()))

    return 1 if class_dict[str(predict_cla)] == "Neoclassism" else 0


def _classification_text(probs: torch.Tensor, class_dict: dict) -> str:
    """把一张图的 softmax 概率格式化成 result.txt 中的一行。"""
    predict_cla = int(torch.argmax(probs).item())
    name = class_dict[str(predict_cla)]
    text = "--->class: {}   prob: {:.3}".format(name, probs[predict_cla].item())
    style = IMPLICIT_STYLES.get(name)
    if style:
        return text + "   implicit style: " + style
    return text


def element_classification(img_path: str, device: torch.device, data_root: str) -> str:
    """单张裁剪图的元素风格分类，返回写入 result.txt 的文本。"""
    model, class_dict = _CACHE.classifier(data_root, device)
    img = _to_tensor(Image.open(img_path), _CLASSIFY_TRANSFORM)
    probs = _softmax_predict(model, img.to(device))[0]
    return _classification_text(probs, class_dict)


def classify_images(images: Sequence[Image.Image], device: torch.device,
                    data_root: str) -> List[str]:
    """批量分类多张裁剪图，返回与输入顺序一致的文本列表。

    一次前向即可处理同类的所有裁剪图，避免逐张推理的开销。
    """
    if not images:
        return []
    model, class_dict = _CACHE.classifier(data_root, device)
    batch = torch.cat([_to_tensor(img, _CLASSIFY_TRANSFORM) for img in images], dim=0)
    probs = _softmax_predict(model, batch.to(device))
    return [_classification_text(p, class_dict) for p in probs]


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


def element_detection(img_path: str, device: torch.device, save_dir: str, txt_path: str,
                      data_root: str, task: int, line_thickness: int,
                      font_size: int) -> List[list]:
    """执行一个阶段的元素检测，返回 [标签, 裁剪图路径] 列表。"""
    config = TASK_CONFIGS[task]

    img_save_dir = os.path.join(save_dir, config.image_dir)
    os.makedirs(img_save_dir, exist_ok=True)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(config.txt_header)

    model, class_dict = _CACHE.detector(data_root, device)
    category_index = {v: k for k, v in class_dict.items()}

    original_img = Image.open(img_path).convert("RGB")
    img = _to_tensor(original_img, _DETECT_TRANSFORM)

    with torch.no_grad():
        predictions = model(img.to(device))[0]

    predict_boxes, predict_classes, predict_scores = _split_pediment(predictions, category_index)

    crop_imgs, string_list, labels, scores = get_element_images(
        original_img, predict_boxes, predict_classes, predict_scores,
        category_index, thresh=config.thresh)

    if not labels:
        print("No object detected!")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("   No object detected!\n")
        return []

    draw_box(original_img,
             predict_boxes, predict_classes, predict_scores, category_index,
             thresh=config.thresh,
             line_thickness=line_thickness,
             font_size=font_size)
    original_img.save(os.path.join(save_dir, config.task_string + ".jpg"))

    for crop, name in zip(crop_imgs, string_list):
        crop.save(os.path.join(img_save_dir, name + ".jpg"))

    # 同一子分类器的裁剪图合并成一次前向；文本仍按检测顺序写回
    texts: List[str] = [""] * len(crop_imgs)
    groups: "OrderedDict[Optional[str], List[int]]" = OrderedDict()
    for index, label in enumerate(labels):
        groups.setdefault(CLASSIFIER_DIRS.get(label), []).append(index)

    for classifier_dir, indices in groups.items():
        if classifier_dir is None:
            continue
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(data_root)), classifier_dir)
        try:
            results = classify_images([crop_imgs[i] for i in indices], device, weights_dir)
        except FileNotFoundError as exc:
            print(f"[warn] 跳过 {classifier_dir} 风格细分: {exc}")
            continue
        for index, text in zip(indices, results):
            texts[index] = text

    return_info = []
    with open(txt_path, "a", encoding="utf-8") as f:
        for name, label, score, text, crop in zip(string_list, labels, scores, texts, crop_imgs):
            crop_path = os.path.join(img_save_dir, name + ".jpg")
            f.write("    ")
            f.write(name)
            f.write("    ")
            f.write("score:" + score)
            f.write("    ")
            f.write(text)
            f.write("    ")
            f.write("--->img_path:" + crop_path)
            f.write("\n")

            if label in ("colonnade", "arcade"):
                return_info.append([label, crop_path])

    return return_info


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
