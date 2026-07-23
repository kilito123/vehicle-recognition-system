"""
车辆识别系统 - 全局配置
基于模块化AI系统设计思想，参考Fable/Mythos框架理念
"""
import os

# ============================================
# 基础路径配置
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保目录存在
for d in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================
# YOLO 模型配置 - 使用COCO预训练权重检测车辆
# ============================================
# COCO数据集包含的车辆相关类别：
# 2=car, 3=motorcycle, 5=bus, 7=truck
YOLO_CONFIG = {
    "weights_path": os.path.join(MODELS_DIR, "yolov3.weights"),
    "config_path": os.path.join(MODELS_DIR, "yolov3.cfg"),
    "names_path": os.path.join(MODELS_DIR, "coco.names"),
    "input_size": (416, 416),
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    # COCO中感兴趣的类别索引
    "vehicle_classes": [2, 3, 5, 7],
}

# ============================================
# 车牌检测配置（自定义训练模型）
# ============================================
PLATE_CONFIG = {
    "weights_path": os.path.join(MODELS_DIR, "yolov3_training_final.weights"),
    "config_path": os.path.join(MODELS_DIR, "yolov3_training.cfg"),
    "classes": ["number_plate"],
    "input_size": (416, 416),
    "confidence_threshold": 0.3,
    "nms_threshold": 0.4,
}

# ============================================
# OCR 配置
# ============================================
OCR_CONFIG = {
    # Tesseract路径 - 根据不同系统自动检测
    "tesseract_paths": [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Windows
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",  # Windows x86
        "/usr/bin/tesseract",  # Linux
        "/opt/homebrew/bin/tesseract",  # macOS ARM
        "/usr/local/bin/tesseract",  # macOS Intel
    ],
    # 语言设置 - 中文车牌用chi_sim，英文/数字用eng
    "lang": "eng+chi_sim",
    # OCR配置参数
    "config": "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
}

# ============================================
# 车辆分类配置
# ============================================
VEHICLE_CONFIG = {
    # 车辆类型映射（COCO类别索引 -> 中文名称）
    "class_names": {
        2: ("car", "轿车/小汽车"),
        3: ("motorcycle", "摩托车"),
        5: ("bus", "公交车"),
        7: ("truck", "卡车/货车"),
    },
    # 颜色检测的HSV范围
    "color_ranges": {
        "黑色": [(0, 0, 0), (180, 255, 50)],
        "白色": [(0, 0, 200), (180, 30, 255)],
        "灰色/银色": [(0, 0, 50), (180, 30, 200)],
        "红色": [(0, 100, 100), (10, 255, 255)],
        "深红色": [(160, 100, 100), (180, 255, 255)],
        "橙色": [(10, 100, 100), (25, 255, 255)],
        "黄色": [(25, 100, 100), (35, 255, 255)],
        "绿色": [(35, 100, 100), (85, 255, 255)],
        "蓝色": [(85, 100, 100), (125, 255, 255)],
        "紫色": [(125, 100, 100), (160, 255, 255)],
    },
}

# ============================================
# 视频处理配置
# ============================================
VIDEO_CONFIG = {
    "codec": "mp4v",  # 视频编码器
    "fps": 30,
    "resize_factor": 0.5,  # 处理时的缩放比例
}

# ============================================
# Web应用配置
# ============================================
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "upload_folder": os.path.join(BASE_DIR, "web", "static", "uploads"),
    "max_content_length": 16 * 1024 * 1024,  # 16MB
    "allowed_extensions": {"png", "jpg", "jpeg", "gif", "mp4", "avi", "mov"},
}

# ============================================
# 可视化配置
# ============================================
VISUAL_CONFIG = {
    # 各类别的显示颜色 (BGR格式)
    "colors": {
        "car": (0, 255, 0),          # 绿色
        "motorcycle": (0, 165, 255), # 橙色
        "bus": (255, 0, 0),          # 蓝色
        "truck": (0, 0, 255),        # 红色
        "number_plate": (255, 0, 255), # 紫色
    },
    "font": 0,  # cv2.FONT_HERSHEY_SIMPLEX
    "font_scale": 0.6,
    "thickness": 2,
    "box_thickness": 2,
}
