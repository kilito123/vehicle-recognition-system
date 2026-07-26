"""
车牌检测模块
基于YOLO自定义训练模型，专门检测车牌区域
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLATE_CONFIG, VISUAL_CONFIG


class PlateDetector:
    """
    车牌检测器
    专门用于检测图像中的车牌区域
    """

    def __init__(self, weights_path=None, config_path=None):
        """
        初始化车牌检测器
        Args:
            weights_path: 车牌检测模型权重路径
            config_path: 车牌检测模型配置路径
        """
        self.weights_path = weights_path or PLATE_CONFIG["weights_path"]
        self.config_path = config_path or PLATE_CONFIG["config_path"]
        self.classes = PLATE_CONFIG["classes"]
        self.confidence_threshold = PLATE_CONFIG["confidence_threshold"]
        self.nms_threshold = PLATE_CONFIG["nms_threshold"]
        self.input_size = PLATE_CONFIG["input_size"]

        self.net = None
        self.output_layers = None
        self._load_network()

    def _load_network(self):
        """加载车牌检测网络"""
        if not os.path.exists(self.weights_path):
            print(f"警告: 车牌检测权重文件不存在: {self.weights_path}")
            print("请提供训练好的车牌检测权重文件")
            return

        if not os.path.exists(self.config_path):
            print(f"警告: 车牌检测配置文件不存在: {self.config_path}")
            return

        try:
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            if isinstance(unconnected[0], (list, np.ndarray)):
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            print("车牌检测模型加载成功")
        except Exception as e:
            print(f"车牌检测模型加载失败: {e}")
            self.net = None

    def detect(self, image):
        """
        检测图像中的车牌
        Args:
            image: 输入图像
        Returns:
            list: 车牌检测结果列表
                {
                    'bbox': (x, y, w, h),
                    'confidence': float,
                    'plate_img': 裁剪出的车牌图像
                }
        """
        if self.net is None:
            return []

        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if img is None:
            return []

        height, width = img.shape[:2]

        # 创建blob
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, self.input_size,
            (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.confidence_threshold,
            self.nms_threshold
        )

        results = []
        if len(indices) > 0:
            indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in indices:
                x, y, w, h = boxes[i]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)

                plate_img = img[y1:y2, x1:x2]

                results.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidences[i],
                    'plate_img': plate_img,
                    'bbox_abs': (x1, y1, x2, y2)
                })

        return results

    def detect_in_region(self, image, region_bbox):
        """
        在指定区域内检测车牌（如在已检测到的车辆区域内）
        Args:
            image: 完整图像
            region_bbox: 区域边界框 (x1, y1, x2, y2)
        Returns:
            车牌检测结果列表（坐标相对于原图）
        """
        x1, y1, x2, y2 = region_bbox
        region = image[y1:y2, x1:x2]

        plates = self.detect(region)

        # 调整坐标为原图坐标
        for plate in plates:
            px, py, pw, ph = plate['bbox']
            plate['bbox'] = (px + x1, py + y1, pw, ph)
            pxa, pya, pxb, pyb = plate['bbox_abs']
            plate['bbox_abs'] = (pxa + x1, pya + y1, pxb + x1, pyb + y1)

        return plates

    def draw_detections(self, image, detections):
        """绘制车牌检测框"""
        img = image.copy()
        color = VISUAL_CONFIG["colors"].get("number_plate", (255, 0, 255))
        thickness = VISUAL_CONFIG["thickness"]
        font = VISUAL_CONFIG["font"]

        for det in detections:
            x, y, w, h = det['bbox']
            conf = det['confidence']

            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            text = f"车牌 {conf:.2%}"
            (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
            cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y), color, -1)
            cv2.putText(img, text, (x + 2, y - 3), font, 0.5, (255, 255, 255), 1)

        return img
