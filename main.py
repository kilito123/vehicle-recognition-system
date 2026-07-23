#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆识别系统 - 命令行主程序

使用方法:
    python main.py --image path/to/image.jpg          # 单张图像
    python main.py --video path/to/video.mp4          # 视频文件
    python main.py --dir path/to/images/              # 批量处理目录
    python main.py --web                              # 启动Web界面
    python main.py --camera                           # 摄像头实时检测

示例:
    python main.py --image data/input/car.jpg
    python main.py --video data/input/traffic.mp4 --output output/
    python main.py --dir data/input/ --output data/output/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.detector import UnifiedDetector
from config import INPUT_DIR, OUTPUT_DIR, WEB_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description='车辆识别系统 - 检测车辆、识别车牌',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --image car.jpg                    # 识别单张图像
  %(prog)s --video traffic.mp4 --output out/  # 处理视频
  %(prog)s --dir images/ --output results/    # 批量处理
  %(prog)s --web                              # 启动Web界面
  %(prog)s --camera                           # 摄像头实时检测
        """
    )

    # 输入选项（使用 --photo，互斥组中至少要有一个）
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--photo', '-i', type=str,
                             help='输入图像路径')
    input_group.add_argument('--video', type=str,
                             help='输入视频路径')
    input_group.add_argument('--dir', '-d', type=str,
                             help='输入图像目录')
    input_group.add_argument('--web', '-w', action='store_true',
                             help='启动Web界面')
    input_group.add_argument('--camera', '-c', action='store_true',
                             help='使用摄像头实时检测')

    # 输出选项
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存输出结果')

    # 模型选项
    parser.add_argument('--vehicle-weights', type=str,
                        help='车辆检测模型权重路径')
    parser.add_argument('--vehicle-config', type=str,
                        help='车辆检测模型配置路径')
    parser.add_argument('--plate-weights', type=str,
                        help='车牌检测模型权重路径')
    parser.add_argument('--plate-config', type=str,
                        help='车牌检测模型配置路径')
    parser.add_argument('--tesseract', type=str,
                        help='Tesseract可执行文件路径')

    # 视频选项
    parser.add_argument('--display', action='store_true',
                        help='实时显示处理过程')

    args = parser.parse_args()

    # 初始化检测器
    print("\n" + "=" * 60)
    print("     车辆识别系统 v1.0")
    print("     Vehicle Recognition System")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 检查是否提供了至少一个输入
    if not (args.photo or args.video or args.dir or args.web or args.camera):
        print("\n错误: 请提供至少一个输入参数")
        print("用法示例:")
        print("  python main.py --photo image.jpg")
        print("  python main.py --video video.mp4")
        print("  python main.py --web")
        print("  python main.py --camera")
        sys.exit(1)

    if args.web:
        # 启动Web界面
        print("\n启动Web界面...")
        from web.app import create_app
        app = create_app()
        app.run(
            host=WEB_CONFIG["host"],
            port=WEB_CONFIG["port"],
            debug=WEB_CONFIG["debug"]
        )
        return

    # 初始化检测器
    detector = UnifiedDetector(
        vehicle_weights=args.vehicle_weights,
        vehicle_config=args.vehicle_config,
        plate_weights=args.plate_weights,
        plate_config=args.plate_config,
        tesseract_path=args.tesseract
    )

    if args.photo:
        # 单张图像
        if not os.path.exists(args.photo):
            print(f"错误: 图像不存在: {args.photo}")
            sys.exit(1)

        result = detector.process_image(
            args.photo,
            save_output=not args.no_save,
            output_dir=args.output
        )

        # 显示结果
        print("\n" + "=" * 40)
        print("识别结果:")
        print("=" * 40)
        from utils.helpers import format_detection_result
        print(format_detection_result(result))

        # 显示图像
        if args.display:
            cv2 = __import__('cv2')
            cv2.imshow("Result", result['annotated_image'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.video:
        # 视频处理
        if not os.path.exists(args.video):
            print(f"错误: 视频不存在: {args.video}")
            sys.exit(1)

        output_video = os.path.join(
            args.output,
            f"output_{os.path.basename(args.video)}"
        )
        detector.process_video(
            args.video,
            output_path=output_video if not args.no_save else None,
            display=args.display
        )

    elif args.dir:
        # 批量处理
        if not os.path.isdir(args.dir):
            print(f"错误: 目录不存在: {args.dir}")
            sys.exit(1)

        detector.process_directory(args.dir, output_dir=args.output)

    elif args.camera:
        # 摄像头实时检测
        print("\n启动摄像头实时检测...")
        print("按 'q' 退出")
        run_camera_detection(detector)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


def run_camera_detection(detector):
    """摄像头实时检测"""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测车辆
        vehicles = detector.vehicle_detector.detect(frame)
        annotated = detector.vehicle_detector.draw_detections(frame, vehicles)

        # 添加状态信息
        cv2.putText(annotated, f"Vehicles: {len(vehicles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Vehicle Recognition - Camera", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
