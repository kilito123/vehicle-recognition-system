"""
Flask Web应用 - 车辆识别系统Web界面
提供友好的图形界面进行车辆检测和识别
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.detector import UnifiedDetector
from config import WEB_CONFIG, OUTPUT_DIR


def create_app():
    """创建Flask应用"""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    app.secret_key = 'vehicle_recognition_secret_key_2024'
    app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG['max_content_length']
    app.config['UPLOAD_FOLDER'] = WEB_CONFIG['upload_folder']

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 全局检测器实例
    detector = None

    def get_detector():
        nonlocal detector
        if detector is None:
            detector = UnifiedDetector()
        return detector

    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')

    @app.route('/api/detect', methods=['POST'])
    def api_detect():
        """API: 检测图像中的车辆"""
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图像'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        file.save(file_path)

        try:
            # 执行检测
            det = get_detector()
            result = det.process_image(file_path, save_output=False)

            # 保存标注图像
            annotated_name = f"annotated_{saved_name}"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
            cv2.imwrite(annotated_path, result['annotated_image'])

            # 构建响应
            response = {
                'success': True,
                'original_image': f'/uploads/{saved_name}',
                'annotated_image': f'/uploads/{annotated_name}',
                'vehicle_count': result['vehicle_count'],
                'total_plates': result['total_plates'],
                'recognized_plates': result['recognized_plates'],
                'vehicles': []
            }

            for v in result['vehicles']:
                vehicle_data = {
                    'id': v['vehicle_id'],
                    'type': v['vehicle_type'],
                    'color': v['vehicle_color'],
                    'confidence': v['detection_confidence'],
                    'plates': v['plates']
                }
                response['vehicles'].append(vehicle_data)

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/detect_camera', methods=['POST'])
    def api_detect_camera():
        """API: 处理摄像头捕获的帧"""
        if 'frame' not in request.files:
            return jsonify({'error': '没有收到帧数据'}), 400

        file = request.files['frame']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{timestamp}.jpg")
        file.save(file_path)

        try:
            det = get_detector()
            result = det.process_image(file_path, save_output=False)

            # 简化响应
            response = {
                'success': True,
                'vehicle_count': result['vehicle_count'],
                'vehicles': []
            }

            for v in result['vehicles']:
                response['vehicles'].append({
                    'id': v['vehicle_id'],
                    'type': v['vehicle_type']['subtype'],
                    'color': v['vehicle_color']['primary_color'],
                    'plates': [p['plate_text'] for p in v['plates'] if p['plate_text']]
                })

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/stats')
    def api_stats():
        """API: 获取系统统计信息"""
        return jsonify({
            'system': 'Vehicle Recognition System v1.0',
            'status': 'running',
            'features': [
                '车辆检测 (YOLO)',
                '车牌检测 (YOLO)',
                '车牌OCR (Tesseract)',
                '车辆颜色识别',
                '车辆类型分类'
            ]
        })

    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        """提供上传文件的访问"""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/detect', methods=['GET', 'POST'])
    def detect_page():
        """检测页面（表单上传）"""
        if request.method == 'POST':
            if 'image' not in request.files:
                return render_template('index.html', error='请上传图像')

            file = request.files['image']
            if file.filename == '':
                return render_template('index.html', error='未选择文件')

            # 保存并处理
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_name = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
            file.save(file_path)

            try:
                det = get_detector()
                result = det.process_image(file_path, save_output=False)

                # 保存标注图像
                annotated_name = f"annotated_{saved_name}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
                cv2.imwrite(annotated_path, result['annotated_image'])

                return render_template('results.html',
                                       original=f'/uploads/{saved_name}',
                                       annotated=f'/uploads/{annotated_name}',
                                       result=result)

            except Exception as e:
                return render_template('index.html', error=str(e))

        return render_template('index.html')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host=WEB_CONFIG['host'], port=WEB_CONFIG['port'], debug=WEB_CONFIG['debug'])
