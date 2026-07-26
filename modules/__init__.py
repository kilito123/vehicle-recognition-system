"""
车辆识别系统 - 核心模块包
"""
from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector
from .plate_recognizer import PlateRecognizer
from .vehicle_classifier import VehicleClassifier
from .detector import UnifiedDetector

__all__ = [
    "VehicleDetector",
    "PlateDetector",
    "PlateRecognizer",
    "VehicleClassifier",
    "UnifiedDetector",
]
