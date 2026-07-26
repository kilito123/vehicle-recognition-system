"""
车辆特征分类模块
分析车辆的视觉特征：颜色、类型、大小等
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VEHICLE_CONFIG


class VehicleClassifier:
    """
    车辆分类器
    分析车辆图像，提取颜色、大小等特征
    """

    def __init__(self):
        self.color_ranges = VEHICLE_CONFIG["color_ranges"]

    def classify_color(self, vehicle_img):
        """
        识别车辆主体颜色
        Args:
            vehicle_img: 车辆图像
        Returns:
            dict: 颜色分析结果
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return {
                'primary_color': '未知',
                'color_confidence': 0.0,
                'color_distribution': {}
            }

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)

        # 排除过暗（阴影）和过亮（高光）区域
        mask = cv2.inRange(hsv, (0, 20, 30), (180, 255, 230))

        # 统计各颜色占比
        color_scores = {}
        total_pixels = cv2.countNonZero(mask)

        if total_pixels == 0:
            total_pixels = hsv.shape[0] * hsv.shape[1]
            mask = np.ones((hsv.shape[0], hsv.shape[1]), dtype=np.uint8) * 255

        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # 处理红色（跨越0度）
            if color_name == "红色":
                mask_red1 = cv2.inRange(hsv, lower, upper)
                mask_red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
                color_mask = cv2.bitwise_or(mask_red1, mask_red2)
            else:
                color_mask = cv2.inRange(hsv, lower, upper)

            # 与有效像素区域取交集
            color_mask = cv2.bitwise_and(color_mask, mask)

            pixel_count = cv2.countNonZero(color_mask)
            ratio = pixel_count / total_pixels
            color_scores[color_name] = ratio

        # 找出占比最高的颜色
        if color_scores:
            primary_color = max(color_scores, key=color_scores.get)
            confidence = color_scores[primary_color]
        else:
            primary_color = '未知'
            confidence = 0.0

        # 过滤掉占比过低的颜色
        color_distribution = {
            k: round(v, 4) for k, v in color_scores.items() if v > 0.05
        }

        return {
            'primary_color': primary_color,
            'color_confidence': round(confidence, 4),
            'color_distribution': color_distribution
        }

    def classify_vehicle_type(self, detection_info, vehicle_img):
        """
        进一步细分车辆类型
        基于检测信息和图像特征
        """
        class_name = detection_info.get('class_name', 'unknown')
        bbox = detection_info.get('bbox', (0, 0, 0, 0))
        _, _, w, h = bbox

        # 宽高比分析
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h

        vehicle_type = {
            'base_type': class_name,
            'subtype': class_name,
            'aspect_ratio': round(aspect_ratio, 2),
            'estimated_size': 'medium',
        }

        if class_name == 'car':
            if aspect_ratio > 1.8:
                vehicle_type['subtype'] = '跑车/敞篷车'
            elif aspect_ratio > 1.5:
                vehicle_type['subtype'] = '轿车'
            else:
                vehicle_type['subtype'] = 'SUV/MPV'

        elif class_name == 'truck':
            if aspect_ratio > 2.5:
                vehicle_type['subtype'] = '大型货车/集装箱车'
            elif aspect_ratio > 1.8:
                vehicle_type['subtype'] = '中型货车'
            else:
                vehicle_type['subtype'] = '小型货车/皮卡'

        elif class_name == 'bus':
            if aspect_ratio > 3.0:
                vehicle_type['subtype'] = '长途客车/大巴'
            else:
                vehicle_type['subtype'] = '公交车'

        elif class_name == 'motorcycle':
            vehicle_type['subtype'] = '摩托车'

        # 估计尺寸
        if area < 50000:
            vehicle_type['estimated_size'] = 'small'
        elif area < 150000:
            vehicle_type['estimated_size'] = 'medium'
        else:
            vehicle_type['estimated_size'] = 'large'

        return vehicle_type

    def analyze_vehicle(self, detection_info, vehicle_img):
        """
        综合分析车辆特征
        """
        color_info = self.classify_color(vehicle_img)
        type_info = self.classify_vehicle_type(detection_info, vehicle_img)

        return {
            'color': color_info,
            'type': type_info,
            'detection': detection_info,
        }

    def draw_analysis(self, image, analysis, position=None):
        """
        在图像上绘制分析结果
        """
        img = image.copy()

        if position is None:
            h, w = img.shape[:2]
            position = (10, h - 100)

        x, y = position
        color = (0, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # 颜色信息
        color_info = analysis.get('color', {})
        primary_color = color_info.get('primary_color', '未知')
        color_conf = color_info.get('color_confidence', 0)

        texts = [
            f"颜色: {primary_color} ({color_conf:.1%})",
        ]

        # 类型信息
        type_info = analysis.get('type', {})
        subtype = type_info.get('subtype', '未知')
        texts.append(f"类型: {subtype}")

        # 绘制信息
        line_height = 20
        for i, text in enumerate(texts):
            cv2.putText(img, text, (x, y + i * line_height),
                        font, font_scale, color, thickness)

        return img
