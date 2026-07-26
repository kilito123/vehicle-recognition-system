"""
图像预处理模块
提供车牌和车辆图像的标准化预处理流程
"""
import cv2
import numpy as np


class ImageProcessor:
    """图像预处理工具类，提供多种预处理方法"""

    @staticmethod
    def read_image(image_path):
        """安全读取图像"""
        if not isinstance(image_path, str):
            return image_path  # 已经是图像数组
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return img

    @staticmethod
    def resize_image(img, target_size=None, scale=None):
        """调整图像大小"""
        if target_size:
            return cv2.resize(img, target_size)
        if scale:
            return cv2.resize(img, None, fx=scale, fy=scale)
        return img

    @staticmethod
    def to_grayscale(img):
        """转换为灰度图"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def apply_gaussian_blur(img, kernel_size=(5, 5)):
        """高斯模糊去噪"""
        return cv2.GaussianBlur(img, kernel_size, 0)

    @staticmethod
    def apply_median_blur(img, kernel_size=5):
        """中值滤波去噪（对椒盐噪声效果好）"""
        return cv2.medianBlur(img, kernel_size)

    @staticmethod
    def invert(img):
        """颜色反转"""
        return cv2.bitwise_not(img)

    @staticmethod
    def binarize(img, threshold=127, max_val=255, method=cv2.THRESH_BINARY):
        """二值化"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, threshold, max_val, method)
        return binary

    @staticmethod
    def adaptive_binarize(img, block_size=11, c=2):
        """自适应二值化（适合光照不均的图像）"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )

    @staticmethod
    def dilate(img, kernel_size=(3, 3), iterations=1):
        """膨胀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.dilate(img, kernel, iterations=iterations)

    @staticmethod
    def erode(img, kernel_size=(3, 3), iterations=1):
        """腐蚀操作"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.erode(img, kernel, iterations=iterations)

    @staticmethod
    def open_morphology(img, kernel_size=(3, 3)):
        """开运算（先腐蚀后膨胀，去除小噪声）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def close_morphology(img, kernel_size=(3, 3)):
        """闭运算（先膨胀后腐蚀，填补小空洞）"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def sharpen(img):
        """图像锐化"""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def enhance_contrast(img):
        """增强对比度（CLAHE）"""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(img)

    @staticmethod
    def normalize_lighting(img):
        """光照归一化"""
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            normalized = cv2.merge([l, a, b])
            return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(img)

    @staticmethod
    def deskew_plate(img):
        """车牌倾斜校正"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        # 找到最大的轮廓
        largest = max(contours, key=cv2.contourArea)

        # 获取最小外接矩形和角度
        rect = cv2.minAreaRect(largest)
        angle = rect[-1]

        # 校正角度
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 5:  # 只有明显倾斜时才校正
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
            return rotated

        return img

    @staticmethod
    def plate_preprocessing_pipeline(img):
        """
        车牌专用预处理流水线
        针对车牌OCR优化的完整预处理流程
        """
        # 1. 光照归一化
        img = ImageProcessor.normalize_lighting(img)

        # 2. 倾斜校正
        img = ImageProcessor.deskew_plate(img)

        # 3. 转灰度
        gray = ImageProcessor.to_grayscale(img)

        # 4. 高斯模糊去噪
        blurred = ImageProcessor.apply_gaussian_blur(gray, (3, 3))

        # 5. 自适应二值化
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 6. 形态学闭运算填补字符空洞
        closed = ImageProcessor.close_morphology(binary, (2, 2))

        return closed

    @staticmethod
    def vehicle_preprocessing_pipeline(img):
        """
        车辆图像预处理流水线
        用于车辆特征分析的预处理
        """
        # 1. 调整大小
        img = ImageProcessor.resize_image(img, scale=0.5)

        # 2. 增强对比度
        img = ImageProcessor.enhance_contrast(img)

        # 3. 轻微高斯模糊去噪
        img = ImageProcessor.apply_gaussian_blur(img, (3, 3))

        return img


# 便捷函数接口
def preprocess_for_ocr(img):
    """为OCR预处理的便捷函数"""
    return ImageProcessor.plate_preprocessing_pipeline(img)


def preprocess_for_detection(img):
    """为检测预处理的便捷函数"""
    return ImageProcessor.vehicle_preprocessing_pipeline(img)
