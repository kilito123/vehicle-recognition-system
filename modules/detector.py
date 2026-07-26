"""
统一检测管道模块
整合车辆检测、车牌检测、OCR识别的完整流水线
参考Fable/Mythos的模块化设计理念
"""
import cv2
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.vehicle_detector import VehicleDetector
from modules.plate_detector import PlateDetector
from modules.plate_recognizer import PlateRecognizer
from modules.vehicle_classifier import VehicleClassifier
from config import OUTPUT_DIR, VISUAL_CONFIG


class UnifiedDetector:
    """
    统一检测器
    整合所有检测模块，提供完整的车辆识别流水线

    流水线流程:
    输入图像 -> 车辆检测 -> 车辆分类 -> 车牌检测 -> OCR识别 -> 结果输出
    """

    def __init__(self, vehicle_weights=None, vehicle_config=None,
                 plate_weights=None, plate_config=None,
                 tesseract_path=None):
        """
        初始化统一检测器
        可选传入自定义模型路径，否则使用默认配置
        """
        print("=" * 50)
        print("初始化车辆识别系统...")
        print("=" * 50)

        # 初始化各模块
        print("\n[1/4] 加载车辆检测模型...")
        self.vehicle_detector = VehicleDetector(vehicle_weights, vehicle_config)

        print("\n[2/4] 加载车牌检测模型...")
        self.plate_detector = PlateDetector(plate_weights, plate_config)

        print("\n[3/4] 加载OCR识别引擎...")
        self.plate_recognizer = PlateRecognizer(tesseract_path)

        print("\n[4/4] 加载车辆分类器...")
        self.vehicle_classifier = VehicleClassifier()

        print("\n" + "=" * 50)
        print("车辆识别系统初始化完成!")
        print("=" * 50)

    def process_image(self, image_path, save_output=True, output_dir=None):
        """
        处理单张图像
        完整的车辆识别流水线

        Args:
            image_path: 图像路径
            save_output: 是否保存输出
            output_dir: 输出目录

        Returns:
            dict: 完整的识别结果
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        print(f"\n处理图像: {image_path}")
        print("-" * 40)

        # 步骤1: 车辆检测
        print("步骤1: 检测车辆...")
        vehicles = self.vehicle_detector.detect(img)
        print(f"  检测到 {len(vehicles)} 辆车辆")

        if not vehicles:
            return {
                'image_path': image_path,
                'vehicle_count': 0,
                'vehicles': [],
                'annotated_image': img
            }

        # 步骤2: 处理每辆车
        results = []
        annotated_img = img.copy()

        for i, vehicle in enumerate(vehicles):
            print(f"\n  车辆 {i + 1}:")

            # 裁剪车辆区域
            vehicle_img = self.vehicle_detector.crop_detection(img, vehicle)

            # 车辆特征分析
            analysis = self.vehicle_classifier.analyze_vehicle(vehicle, vehicle_img)
            color_info = analysis['color']
            type_info = analysis['type']
            print(f"    类型: {type_info['subtype']} ({type_info['base_type']})")
            print(f"    颜色: {color_info['primary_color']}")

            # 车牌检测（在车辆区域内）
            print(f"    检测车牌...")
            plates = self.plate_detector.detect_in_region(
                img, vehicle['bbox_abs']
            )
            print(f"    发现 {len(plates)} 个车牌区域")

            # 车牌OCR识别
            plate_results = []
            for j, plate in enumerate(plates):
                plate_img = plate['plate_img']
                ocr_result = self.plate_recognizer.recognize(plate_img)

                plate_text = ocr_result['text']
                plate_conf = ocr_result['confidence']
                plate_info = self.plate_recognizer.extract_plate_info(plate_text)

                print(f"      车牌 {j + 1}: {plate_text or '未识别'} "
                      f"(置信度: {plate_conf:.1%})")

                plate_results.append({
                    'plate_text': plate_text,
                    'confidence': plate_conf,
                    'plate_info': plate_info,
                    'bbox': plate['bbox'],
                })

            # 构建车辆结果
            vehicle_result = {
                'vehicle_id': i + 1,
                'vehicle_type': type_info,
                'vehicle_color': color_info,
                'vehicle_bbox': vehicle['bbox'],
                'detection_confidence': vehicle['confidence'],
                'plates': plate_results,
                'plate_count': len(plate_results),
            }
            results.append(vehicle_result)

            # 在图像上绘制结果
            annotated_img = self._draw_vehicle_result(
                annotated_img, vehicle, analysis, plate_results
            )

        # 汇总统计
        total_plates = sum(v['plate_count'] for v in results)
        recognized_plates = sum(
            1 for v in results for p in v['plates'] if p['plate_text']
        )

        final_result = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'vehicle_count': len(vehicles),
            'total_plates': total_plates,
            'recognized_plates': recognized_plates,
            'vehicles': results,
            'annotated_image': annotated_img,
        }

        # 保存输出
        if save_output:
            self._save_results(final_result, output_dir)

        return final_result

    def process_video(self, video_path, output_path=None, display=False):
        """
        处理视频文件
        逐帧检测车辆并输出标注视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            display: 是否实时显示
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_vehicles = 0

        print(f"\n处理视频: {video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}")
        print("-" * 40)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 每隔几帧检测一次（优化性能）
            if frame_count % 3 == 0:
                vehicles = self.vehicle_detector.detect(frame)
                total_vehicles += len(vehicles)

                # 绘制检测结果
                annotated = self.vehicle_detector.draw_detections(frame, vehicles)

                # 添加帧信息
                cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Vehicles: {len(vehicles)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                annotated = frame

            # 写入输出视频
            if writer:
                writer.write(annotated)

            # 显示
            if display:
                cv2.imshow("Vehicle Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 进度显示
            if frame_count % 30 == 0:
                print(f"  已处理 {frame_count} 帧...")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n视频处理完成!")
        print(f"总帧数: {frame_count}")
        print(f"检测到车辆: {total_vehicles} 次")
        if output_path:
            print(f"输出视频: {output_path}")

    def process_directory(self, input_dir, output_dir=None):
        """
        批量处理目录中的所有图像
        """
        import glob

        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"\n发现 {len(image_paths)} 张图像")
        print("=" * 50)

        all_results = []
        for img_path in sorted(image_paths):
            try:
                result = self.process_image(img_path, save_output=True,
                                            output_dir=output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"处理失败 {img_path}: {e}")

        # 保存汇总报告
        self._save_summary_report(all_results, output_dir)

        return all_results

    def _draw_vehicle_result(self, image, vehicle, analysis, plates):
        """
        在图像上绘制车辆和车牌识别结果
        """
        img = image.copy()
        x, y, w, h = vehicle['bbox']

        # 车辆边界框颜色
        class_name = vehicle['class_name']
        color = VISUAL_CONFIG["colors"].get(class_name, (0, 255, 255))

        # 绘制车辆框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # 车辆标签
        type_info = analysis['type']
        color_info = analysis['color']
        label = f"{type_info['subtype']} | {color_info['primary_color']}"
        conf = vehicle['confidence']

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 25), (x + tw + 10, y), color, -1)
        cv2.putText(img, label, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"{conf:.1%}", (x + 5, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制车牌结果
        for plate in plates:
            px, py, pw, ph = plate['bbox']
            plate_text = plate.get('plate_text', '')
            p_conf = plate.get('confidence', 0)

            # 车牌框
            cv2.rectangle(img, (px, py), (px + pw, py + ph),
                          (255, 0, 255), 2)

            # 车牌文字
            if plate_text:
                text = f"{plate_text} ({p_conf:.0%})"
                (ptw, pth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (px, py - pth - 5), (px + ptw + 5, py),
                              (255, 0, 255), -1)
                cv2.putText(img, text, (px + 2, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    def _save_results(self, result, output_dir=None):
        """保存识别结果到文件"""
        out_dir = output_dir or OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存标注图像
        img_name = os.path.basename(result['image_path'])
        name_without_ext = os.path.splitext(img_name)[0]

        annotated_path = os.path.join(
            out_dir, f"{name_without_ext}_annotated_{timestamp}.jpg"
        )
        cv2.imwrite(annotated_path, result['annotated_image'])
        print(f"  已保存标注图像: {annotated_path}")

        # 保存JSON结果
        json_path = os.path.join(
            out_dir, f"{name_without_ext}_result_{timestamp}.json"
        )
        # 移除numpy数组再序列化
        result_copy = {k: v for k, v in result.items() if k != 'annotated_image'}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, ensure_ascii=False, indent=2)
        print(f"  已保存JSON结果: {json_path}")

        # 保存每辆车的裁剪图
        for vehicle in result['vehicles']:
            v_id = vehicle['vehicle_id']
            # 这里可以添加保存车辆裁剪图的逻辑

        return annotated_path, json_path

    def _save_summary_report(self, all_results, output_dir=None):
        """保存批量处理的汇总报告"""
        out_dir = output_dir or OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        total_vehicles = sum(r['vehicle_count'] for r in all_results)
        total_plates = sum(r['total_plates'] for r in all_results)
        total_recognized = sum(r['recognized_plates'] for r in all_results)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(all_results),
            'total_vehicles': total_vehicles,
            'total_plates': total_plates,
            'total_recognized_plates': total_recognized,
            'recognition_rate': round(total_recognized / total_plates, 4)
            if total_plates > 0 else 0,
            'per_image_results': [
                {
                    'image': r['image_path'],
                    'vehicles': r['vehicle_count'],
                    'plates': r['total_plates'],
                    'recognized': r['recognized_plates'],
                }
                for r in all_results
            ]
        }

        report_path = os.path.join(out_dir, 'summary_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n汇总报告已保存: {report_path}")
        print(f"  总图像数: {summary['total_images']}")
        print(f"  总车辆数: {summary['total_vehicles']}")
        print(f"  总车牌数: {summary['total_plates']}")
        print(f"  识别成功: {summary['total_recognized_plates']}")
        print(f"  识别率: {summary['recognition_rate']:.1%}")
