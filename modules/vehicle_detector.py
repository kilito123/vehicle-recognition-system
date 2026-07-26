"""
车辆检测模块
使用YOLO预训练模型(COCO数据集)检测图像中的车辆
"""
import cv2
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_CONFIG, VEHICLE_CONFIG, VISUAL_CONFIG


class VehicleDetector:
    """
    车辆检测器
    基于YOLO模型，支持检测轿车、摩托车、公交车、卡车等
    """

    def __init__(self, weights_path=None, config_path=None, names_path=None):
        """
        初始化车辆检测器
        Args:
            weights_path: 模型权重文件路径
            config_path: 模型配置文件路径
            names_path: 类别名称文件路径
        """
        self.weights_path = weights_path or YOLO_CONFIG["weights_path"]
        self.config_path = config_path or YOLO_CONFIG["config_path"]
        self.names_path = names_path or YOLO_CONFIG.get("names_path")

        self.confidence_threshold = YOLO_CONFIG["confidence_threshold"]
        self.nms_threshold = YOLO_CONFIG["nms_threshold"]
        self.input_size = YOLO_CONFIG["input_size"]
        self.vehicle_classes = set(YOLO_CONFIG["vehicle_classes"])

        # 加载类别名称
        self.class_names = self._load_class_names()

        # 加载网络
        self.net = None
        self.output_layers = None
        self._load_network()

    def _load_class_names(self):
        """加载COCO类别名称"""
        if self.names_path and os.path.exists(self.names_path):
            with open(self.names_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        # 默认COCO 80类
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def _load_network(self):
        """加载YOLO神经网络"""
        if not os.path.exists(self.weights_path):
            print(f"警告: 权重文件不存在: {self.weights_path}")
            print("请下载YOLOv3预训练权重: wget https://pjreddie.com/media/files/yolov3.weights")
            return

        if not os.path.exists(self.config_path):
            print(f"警告: 配置文件不存在: {self.config_path}")
            print("请下载YOLOv3配置文件")
            return

        try:
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            # 兼容不同OpenCV版本
            if isinstance(unconnected[0], (list, np.ndarray)):
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
            else:
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            print("车辆检测模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.net = None

    def detect(self, image):
        """
        检测图像中的车辆
        Args:
            image: 输入图像 (numpy数组或图像路径)
        Returns:
            list: 检测结果列表，每个元素为字典:
                {
                    'bbox': (x, y, w, h),
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'label': str  # 中文标签
                }
        """
        if self.net is None:
            print("模型未加载，无法检测")
            return []

        # 读取图像
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()

        if img is None:
            return []

        height, width = img.shape[:2]

        # 创建blob
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255.0, self.input_size,
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # 解析检测结果
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # 只保留车辆类别
                if class_id not in self.vehicle_classes:
                    continue

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # 非极大值抑制
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.confidence_threshold,
            self.nms_threshold
        )

        # 构建结果
        results = []
        if len(indices) > 0:
            for i in indices.flatten() if hasattr(indices, 'flatten') else indices:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                label_info = VEHICLE_CONFIG["class_names"].get(class_id, (class_name, class_name))

                results.append({
                    'bbox': (max(0, x), max(0, y), w, h),
                    'confidence': confidences[i],
                    'class_id': class_id,
                    'class_name': class_name,
                    'label': label_info[1],  # 中文标签
                    'bbox_abs': (max(0, x), max(0, y), min(width, x + w), min(height, y + h))
                })

        return results

    def draw_detections(self, image, detections, show_confidence=True):
        """
        在图像上绘制检测结果
        Args:
            image: 输入图像
            detections: detect()返回的检测结果列表
            show_confidence: 是否显示置信度
        Returns:
            绘制后的图像
        """
        img = image.copy()
        colors = VISUAL_CONFIG["colors"]
        font = VISUAL_CONFIG["font"]
        font_scale = VISUAL_CONFIG["font_scale"]
        thickness = VISUAL_CONFIG["thickness"]

        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det['class_name']
            label = det['label']
            confidence = det['confidence']

            color = colors.get(class_name, (0, 255, 255))

            # 绘制边界框
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

            # 准备标签文本
            text = f"{label}"
            if show_confidence:
                text += f" {confidence:.2%}"

            # 计算文本大小
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # 绘制标签背景
            cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
            cv2.putText(img, text, (x + 2, y - 5), font, font_scale, (255, 255, 255), thickness)

        return img

    def crop_detection(self, image, detection):
        """
        根据检测结果裁剪出车辆区域
        Args:
            image: 原始图像
            detection: 单个检测结果
        Returns:
            裁剪后的车辆图像
        """
        x1, y1, x2, y2 = detection['bbox_abs']
        return image[y1:y2, x1:x2]

    def get_detection_count(self, image):
        """获取检测到的车辆数量"""
        return len(self.detect(image))

    def detect_stream(self, frame):
        """
        为视频流优化的检测（减少处理）
        Args:
            frame: 视频帧
        Returns:
            检测结果和绘制后的帧
        """
        detections = self.detect(frame)
        annotated = self.draw_detections(frame, detections)
        return detections, annotated
