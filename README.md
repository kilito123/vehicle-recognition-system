# 🚗 Vehicle Recognition System

> **基于深度学习的车辆检测与车牌识别系统** | Deep Learning-based Vehicle Detection & License Plate Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLO-v3-red.svg)](https://pjreddie.com/darknet/yolo/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 功能亮点

| 功能 | 描述 |
|------|------|
| 🔍 **多类型车辆检测** | 基于 YOLO 实时检测轿车、SUV、卡车、公交车、摩托车 |
| 📋 **车牌定位与识别** | 精准定位车牌区域，OCR 提取车牌号码 |
| 🎨 **车辆特征分析** | 自动识别车辆颜色和细分车型 |
| 🎥 **视频流处理** | 支持视频文件、实时摄像头检测 |
| 🌐 **Web 可视化界面** | 现代化网页界面，支持拖拽上传和摄像头检测 |
| 📁 **批量处理** | 支持整个目录的批量图像处理 |

---

## 🎬 演示效果

### 车辆检测与车牌识别
```
┌─────────────────────────────────────────────┐
│  [输入图像]          [检测结果]               │
│                                             │
│  ┌─────────┐       ┌─────────────────────┐  │
│  │         │       │ ┌───────┐ 京A12345  │  │
│  │  🚗🚙  │  →   │ │ 车牌框 │ 轿车|白色 │  │
│  │         │       │ └───────┘ 置信度95% │  │
│  └─────────┘       └─────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

### Web 界面预览
- 📤 **拖拽上传**：支持拖放图像文件
- 📷 **实时摄像头**：浏览器直接调用摄像头检测
- 📊 **结果展示**：详细统计卡片 + 标注图像

---

## 🏗️ 系统架构

```
输入层                    处理层                      输出层
┌─────────┐    ┌─────────────────────────────┐    ┌────────────┐
│ 图像    │ →  │  ① 车辆检测 (YOLO + COCO)   │ →  │ 车辆类型   │
│ 视频    │    │      ↓ 裁剪车辆区域          │    │ 边界框     │
│ 摄像头  │    │  ② 车辆分类 (颜色/子类型)    │    │ 置信度     │
└─────────┘    │      ↓                      │    └────────────┘
               │  ③ 车牌检测 (YOLO 自定义)    │         ↓
               │      ↓ 裁剪车牌区域          │    ┌────────────┐
               │  ④ 图像预处理 (OpenCV)       │ →  │ 车牌号码   │
               │      ↓                      │    │ OCR置信度  │
               │  ⑤ OCR识别 (Tesseract)      │    │ 省份信息   │
               └─────────────────────────────┘    └────────────┘
```

---

## 📦 安装

### 环境要求

- Python 3.8+
- Windows / macOS / Linux

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/vehicle-recognition-system.git
cd vehicle-recognition-system
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 Tesseract OCR

| 系统 | 安装方式 |
|------|---------|
| **Windows** | [下载安装包](https://github.com/UB-Mannheim/tesseract/wiki) → 安装到默认路径 |
| **macOS** | `brew install tesseract tesseract-lang` |
| **Linux** | `sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim` |

### 4. 下载 YOLO 模型权重

```bash
cd models/

# 下载 YOLOv3 COCO 预训练权重 (~236MB)
curl -L -O https://pjreddie.com/media/files/yolov3.weights

# 下载配置文件
curl -L -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

> 💡 **提示**：如果 `curl` 不可用，可直接用浏览器访问上述链接下载，然后放到 `models/` 目录。

### 5. 配置车牌检测模型（可选）

如果你有训练好的车牌检测权重，放入 `models/` 目录：
- `models/yolov3_training_final.weights` — 车牌检测权重
- `models/yolov3_training.cfg` — 车牌检测配置

---

## 🚀 快速开始

### 命令行使用

```bash
# 单张图像检测
python main.py --photo image.jpg --display

# 视频处理
python main.py --video traffic.mp4 --output results/

# 批量处理目录
python main.py --dir images/ --output results/

# 摄像头实时检测
python main.py --camera

# 启动 Web 界面
python main.py --web
```

### Python API

```python
from modules.detector import UnifiedDetector

# 初始化检测器
detector = UnifiedDetector()

# 处理单张图像
result = detector.process_image("path/to/image.jpg")

# 打印结果
print(f"检测到 {result['vehicle_count']} 辆车")
for v in result['vehicles']:
    print(f"  类型: {v['vehicle_type']['subtype']}")
    print(f"  颜色: {v['vehicle_color']['primary_color']}")
    for p in v['plates']:
        print(f"  车牌: {p['plate_text']}")

# 处理视频
detector.process_video("video.mp4", output_path="output.mp4")

# 批量处理
detector.process_directory("images/", output_dir="results/")
```

### Web 界面

```bash
python main.py --web
```

然后浏览器访问：**http://localhost:5000**

---

## 📁 项目结构

```
vehicle-recognition-system/
├── 📄 main.py                      # 命令行入口
├── 📄 config.py                    # 全局配置
├── 📄 video_detection.py           # 视频处理模块
├── 📄 requirements.txt             # Python依赖
│
├── 🧠 modules/                     # 核心AI模块
│   ├── detector.py                # ⭐ 统一检测管道
│   ├── vehicle_detector.py        # 车辆检测 (YOLO)
│   ├── plate_detector.py          # 车牌检测 (YOLO)
│   ├── plate_recognizer.py        # 车牌OCR (Tesseract)
│   ├── vehicle_classifier.py      # 车辆特征分析
│   └── image_processor.py         # 图像预处理工具
│
├── 🌐 web/                         # Web界面
│   ├── app.py                     # Flask后端
│   ├── templates/                 # HTML模板
│   └── static/                    # CSS/JS资源
│
├── 🛠️ utils/                       # 工具函数
│   └── helpers.py
│
├── 🗂️ models/                      # 模型文件目录
│   ├── coco.names                 # COCO类别名称
│   ├── yolov3.weights             # YOLOv3权重 (需下载)
│   ├── yolov3.cfg                 # YOLOv3配置 (需下载)
│   └── yolov3_training_final.weights  # 车牌权重 (可选)
│
└── 📂 data/                        # 数据目录
    ├── input/                     # 输入图像
    └── output/                    # 输出结果
```

---

## 🔧 配置说明

所有配置集中在 [`config.py`](config.py)，主要参数：

```python
# 车辆检测阈值
YOLO_CONFIG = {
    "confidence_threshold": 0.5,  # 置信度阈值
    "nms_threshold": 0.4,         # NMS IoU阈值
    "vehicle_classes": [2, 3, 5, 7],  # car, motorcycle, bus, truck
}

# OCR配置
OCR_CONFIG = {
    "lang": "eng+chi_sim",        # 识别语言
    "config": "--psm 7 ...",      # Tesseract参数
}

# Web服务
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
}
```

---

## 📊 检测效果示例

```
==================================================
               车辆识别系统 v1.0
==================================================

处理图像: data/input/traffic.jpg
----------------------------------------
步骤1: 检测车辆...
  检测到 3 辆车辆

  车辆 1:
    类型: 轿车 (car)
    颜色: 白色
    检测车牌...
    发现 1 个车牌区域
      车牌 1: 京A12345 (置信度: 85.2%)

  车辆 2:
    类型: SUV/MPV (car)
    颜色: 黑色
    检测车牌...
    发现 1 个车牌区域
      车牌 1: 沪B67890 (置信度: 78.5%)

  车辆 3:
    类型: 公交车 (bus)
    颜色: 蓝色
    检测车牌...
    发现 1 个车牌区域
      车牌 1: 粤C11111 (置信度: 92.1%)
```

---

## 🛠️ 技术栈

| 技术 | 用途 |
|------|------|
| [YOLOv3](https://pjreddie.com/darknet/yolo/) | 实时目标检测 |
| [OpenCV](https://opencv.org/) | 图像处理与预处理 |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | 光学字符识别 |
| [Flask](https://flask.palletsprojects.com/) | Web后端服务 |
| [NumPy](https://numpy.org/) | 数值计算 |

---

## 🎯 支持的检测类别

### 车辆类型
| 类别 | 英文名 | 子类型识别 |
|------|--------|-----------|
| 🚗 轿车 | car | 轿车 / SUV / 跑车 |
| 🚌 公交车 | bus | 公交 / 长途客车 |
| 🚚 卡车 | truck | 小型货车 / 中型货车 / 大型货车 |
| 🏍️ 摩托车 | motorcycle | 摩托车 |

### 颜色识别
支持识别：黑色、白色、灰色、红色、橙色、黄色、绿色、蓝色、紫色

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证开源。

---

## 🙏 致谢

- [YOLO - You Only Look Once](https://pjreddie.com/darknet/yolo/) by Joseph Redmon
- [OpenCV](https://opencv.org/) - 开源计算机视觉库
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Google 开源 OCR 引擎
- [Flask](https://flask.palletsprojects.com/) - Python Web 框架

---

<div align="center">

⭐ **如果这个项目对你有帮助，请点个 Star 支持一下！** ⭐

</div>
