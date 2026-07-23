#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频车辆检测模块
专门用于处理视频文件的车辆检测和跟踪
"""
import cv2
import numpy as np
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.vehicle_detector import VehicleDetector
from modules.plate_detector import PlateDetector
from modules.plate_recognizer import PlateRecognizer
from config import OUTPUT_DIR


class VideoVehicleDetector:
    """
    视频车辆检测器
    支持车辆检测、简单跟踪、统计计数
    """

    def __init__(self, vehicle_weights=None, vehicle_config=None):
        self.vehicle_detector = VehicleDetector(vehicle_weights, vehicle_config)
        self.plate_detector = PlateDetector()
        self.plate_recognizer = PlateRecognizer()

        # 跟踪状态
        self.tracks = {}
        self.next_track_id = 0
        self.track_history = defaultdict(list)

        # 统计
        self.total_count = 0
        self.frame_count = 0

    def detect_in_video(self, video_path, output_path=None,
                        display=True, skip_frames=2):
        """
        处理视频文件
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            display: 是否实时显示
            skip_frames: 跳帧数（每N帧检测一次）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"\n视频信息:")
        print(f"  路径: {video_path}")
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {total_frames / fps:.1f}秒")
        print("-" * 40)

        detections_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # 跳帧优化
            if self.frame_count % (skip_frames + 1) == 0:
                # 检测
                vehicles = self.vehicle_detector.detect(frame)
                detections_per_frame.append(len(vehicles))
                self.total_count += len(vehicles)

                # 绘制
                annotated = self.vehicle_detector.draw_detections(frame, vehicles)

                # 添加统计信息
                self._draw_stats(annotated, vehicles)
            else:
                annotated = frame

            # 写入输出
            if writer:
                writer.write(annotated)

            # 显示
            if display:
                cv2.imshow("Vehicle Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 进度
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100
                print(f"  进度: {progress:.1f}% ({self.frame_count}/{total_frames}帧)")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # 汇总
        avg_vehicles = sum(detections_per_frame) / len(detections_per_frame) \
            if detections_per_frame else 0
        print(f"\n处理完成!")
        print(f"  总帧数: {self.frame_count}")
        print(f"  平均车辆数/帧: {avg_vehicles:.1f}")
        if output_path:
            print(f"  输出视频: {output_path}")

    def _draw_stats(self, image, vehicles):
        """在图像上绘制统计信息"""
        h, w = image.shape[:2]

        # 半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # 统计文字
        cv2.putText(image, f"Frame: {self.frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Current: {len(vehicles)} vehicles", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Total detected: {self.total_count}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def detect_with_plate_tracking(self, video_path, output_path=None,
                                   display=True):
        """
        带车牌跟踪的高级视频检测
        检测车辆并持续跟踪车牌
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 已识别的车牌记录
        recognized_plates = set()
        plate_history = []

        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated = frame.copy()

            # 车辆检测
            vehicles = self.vehicle_detector.detect(frame)

            for v in vehicles:
                x1, y1, x2, y2 = v['bbox_abs']

                # 在车辆区域内检测车牌
                plates = self.plate_detector.detect_in_region(frame, (x1, y1, x2, y2))

                for p in plates:
                    plate_img = p['plate_img']
                    ocr_result = self.plate_recognizer.recognize(plate_img)
                    plate_text = ocr_result['text']

                    if plate_text and plate_text not in recognized_plates:
                        recognized_plates.add(plate_text)
                        plate_history.append({
                            'frame': frame_num,
                            'plate': plate_text,
                            'confidence': ocr_result['confidence'],
                        })
                        print(f"  [帧 {frame_num}] 新识别车牌: {plate_text}")

                    # 绘制车牌框和文字
                    px, py, pw, ph = p['bbox']
                    cv2.rectangle(annotated, (px, py), (px + pw, py + ph),
                                  (255, 0, 255), 2)
                    if plate_text:
                        cv2.putText(annotated, plate_text, (px, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # 绘制车辆框
            annotated = self.vehicle_detector.draw_detections(annotated, vehicles)

            # 添加车牌统计
            cv2.putText(annotated, f"Unique Plates: {len(recognized_plates)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Vehicle + Plate Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # 输出汇总
        print(f"\n视频处理完成!")
        print(f"  识别到 {len(recognized_plates)} 个不同车牌:")
        for p in plate_history:
            print(f"    帧 {p['frame']}: {p['plate']} ({p['confidence']:.1%})")

        return recognized_plates, plate_history


def main():
    parser = argparse.ArgumentParser(description='视频车辆检测')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('-o', '--output', help='输出视频路径')
    parser.add_argument('-d', '--display', action='store_true',
                        help='实时显示')
    parser.add_argument('-p', '--plates', action='store_true',
                        help='启用车牌跟踪')
    parser.add_argument('-s', '--skip', type=int, default=2,
                        help='跳帧数 (默认: 2)')

    args = parser.parse_args()

    detector = VideoVehicleDetector()

    if args.plates:
        detector.detect_with_plate_tracking(
            args.video, args.output, args.display
        )
    else:
        detector.detect_in_video(
            args.video, args.output, args.display, args.skip
        )


if __name__ == '__main__':
    main()
