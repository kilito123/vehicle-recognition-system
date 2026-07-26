"""
工具函数模块
"""
import cv2
import numpy as np
import os


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def get_image_files(directory, extensions=None):
    """获取目录中的所有图像文件"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    files = []
    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return sorted(files)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """保持宽高比调整图像大小"""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def draw_text_with_background(img, text, position, font_scale=0.5,
                               color=(255, 255, 255), bg_color=(0, 0, 0),
                               thickness=1, padding=5):
    """绘制带背景的文本"""
    x, y = position
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)

    # 背景
    cv2.rectangle(img,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + padding),
                  bg_color, -1)

    # 文字
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)

    return img


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def format_detection_result(result):
    """格式化检测结果为可读字符串"""
    lines = []
    lines.append(f"图像: {result.get('image_path', 'N/A')}")
    lines.append(f"检测时间: {result.get('timestamp', 'N/A')}")
    lines.append(f"车辆数量: {result.get('vehicle_count', 0)}")
    lines.append(f"车牌数量: {result.get('total_plates', 0)}")
    lines.append(f"识别成功: {result.get('recognized_plates', 0)}")
    lines.append("")

    for v in result.get('vehicles', []):
        lines.append(f"车辆 #{v['vehicle_id']}:")
        v_type = v.get('vehicle_type', {})
        v_color = v.get('vehicle_color', {})
        lines.append(f"  类型: {v_type.get('subtype', '未知')} "
                     f"({v_type.get('base_type', '未知')})")
        lines.append(f"  颜色: {v_color.get('primary_color', '未知')} "
                     f"({v_color.get('color_confidence', 0):.1%})")
        lines.append(f"  检测置信度: {v.get('detection_confidence', 0):.1%}")

        for p in v.get('plates', []):
            lines.append(f"  车牌: {p.get('plate_text', '未识别')} "
                         f"(OCR置信度: {p.get('confidence', 0):.1%})")
        lines.append("")

    return "\n".join(lines)
