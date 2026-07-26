"""
车牌OCR识别模块
基于Tesseract OCR引擎，识别车牌中的文字
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_CONFIG
from modules.image_processor import ImageProcessor


class PlateRecognizer:
    """
    车牌识别器
    对检测到的车牌图像进行OCR文字识别
    """

    def __init__(self, tesseract_path=None, lang=None):
        """
        初始化车牌识别器
        Args:
            tesseract_path: Tesseract可执行文件路径
            lang: OCR语言设置
        """
        self.lang = lang or OCR_CONFIG["lang"]
        self.config = OCR_CONFIG["config"]
        self._setup_tesseract(tesseract_path)

    def _setup_tesseract(self, custom_path=None):
        """设置Tesseract路径"""
        if custom_path and os.path.exists(custom_path):
            pytesseract.pytesseract.tesseract_cmd = custom_path
            return

        # 自动查找Tesseract
        for path in OCR_CONFIG["tesseract_paths"]:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract路径: {path}")
                return

        print("警告: 未找到Tesseract，请手动安装并设置路径")
        print("Windows下载: https://github.com/UB-Mannheim/tesseract/wiki")

    def _preprocess_plate(self, plate_img):
        """
        车牌图像预处理
        优化OCR识别率
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # 使用专用预处理流水线
        processed = ImageProcessor.plate_preprocessing_pipeline(plate_img)

        # 如果图像太小，放大
        h, w = processed.shape[:2]
        if w < 100:
            scale = 200 / w
            processed = cv2.resize(processed, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

        return processed

    def recognize(self, plate_img, preprocess=True):
        """
        识别车牌文字
        Args:
            plate_img: 车牌图像
            preprocess: 是否进行预处理
        Returns:
            dict: 识别结果
                {
                    'text': 识别出的文字,
                    'confidence': 置信度,
                    'raw_text': 原始识别文本
                }
        """
        if plate_img is None or plate_img.size == 0:
            return {'text': '', 'confidence': 0.0, 'raw_text': ''}

        # 预处理
        if preprocess:
            img = self._preprocess_plate(plate_img)
            if img is None:
                return {'text': '', 'confidence': 0.0, 'raw_text': ''}
        else:
            if len(plate_img.shape) == 3:
                img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                img = plate_img

        # 使用Tesseract进行OCR
        try:
            # 尝试获取详细数据
            data = pytesseract.image_to_data(
                img, lang=self.lang, config=self.config,
                output_type=pytesseract.Output.DICT
            )

            # 提取文字
            raw_text = pytesseract.image_to_string(
                img, lang=self.lang, config=self.config
            )

            # 计算平均置信度
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            # 清理文字
            text = self._clean_plate_text(raw_text)

            return {
                'text': text,
                'confidence': avg_conf / 100.0,
                'raw_text': raw_text.strip(),
            }

        except Exception as e:
            print(f"OCR识别失败: {e}")
            return {'text': '', 'confidence': 0.0, 'raw_text': ''}

    def _clean_plate_text(self, text):
        """
        清理车牌文字
        移除非法字符，保留字母、数字、汉字和连字符
        """
        if not text:
            return ""

        # 移除空白字符
        text = text.strip().replace(' ', '').replace('\n', '')

        # 移除常见误识别字符
        text = text.replace('O', '0').replace('I', '1').replace('Z', '2')

        # 只保留有效字符
        # 中国车牌格式: 省份简称 + 字母 + 数字/字母
        # 保留汉字、字母、数字
        cleaned = re.sub(r'[^一-龥A-Za-z0-9]', '', text)

        return cleaned.upper()

    def recognize_multiple(self, plate_images):
        """
        批量识别多个车牌
        Args:
            plate_images: 车牌图像列表
        Returns:
            识别结果列表
        """
        results = []
        for img in plate_images:
            result = self.recognize(img)
            results.append(result)
        return results

    def recognize_with_visualization(self, plate_img):
        """
        识别车牌并返回可视化结果
        Args:
            plate_img: 车牌图像
        Returns:
            result, visualization
        """
        result = self.recognize(plate_img)

        # 创建可视化图像
        vis = plate_img.copy() if len(plate_img.shape) == 3 else \
            cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

        text = result['text'] or "未识别"
        conf = result['confidence']

        h, w = vis.shape[:2]
        cv2.putText(vis, f"{text}", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.1%}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return result, vis

    def validate_plate_format(self, text):
        """
        验证车牌格式
        支持中国车牌格式验证
        Args:
            text: 车牌文字
        Returns:
            bool: 是否符合基本格式
        """
        if not text or len(text) < 7:
            return False

        # 普通蓝牌/黄牌: 汉字 + 字母 + 5位字母数字
        pattern1 = r'^[一-龥][A-Z][A-Z0-9]{4,6}$'
        # 新能源车牌: 汉字 + 字母 + [DF] + 5位
        pattern2 = r'^[一-龥][A-Z][DF][A-Z0-9]{5}$'

        return bool(re.match(pattern1, text)) or bool(re.match(pattern2, text))

    def extract_plate_info(self, text):
        """
        提取车牌信息
        Args:
            text: 车牌文字
        Returns:
            dict: 车牌信息
        """
        if not text:
            return {}

        info = {
            'plate_number': text,
            'province': '',
            'city_code': '',
            'is_valid': False,
        }

        # 提取省份（第一个汉字）
        for char in text:
            if '一' <= char <= '鿿':
                info['province'] = char
                break

        # 提取城市代码（省份后的字母）
        if len(text) >= 2:
            info['city_code'] = text[1]

        info['is_valid'] = self.validate_plate_format(text)

        return info
