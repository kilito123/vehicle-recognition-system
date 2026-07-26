/**
 * 车辆识别系统 - 前端交互脚本
 */

document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const previewSection = document.getElementById('preview-section');
    const previewImg = document.getElementById('preview-img');
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submit-btn');

    // 点击上传区域触发文件选择
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    // 文件选择后预览
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                previewSection.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });

    // 拖拽上传
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function() {
        this.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            const reader = new FileReader();
            reader.onload = function(ev) {
                previewImg.src = ev.target.result;
                previewSection.style.display = 'block';
            };
            reader.readAsDataURL(files[0]);
        }
    });

    // 表单提交显示加载
    uploadForm.addEventListener('submit', function() {
        submitBtn.disabled = true;
        submitBtn.textContent = '处理中...';
        loading.style.display = 'block';
    });

    // ====== 摄像头功能 ======
    const startCameraBtn = document.getElementById('start-camera');
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const cameraOverlay = document.getElementById('camera-overlay');
    const cameraResults = document.getElementById('camera-results');

    let stream = null;
    let isProcessing = false;

    startCameraBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720 }
            });
            video.srcObject = stream;
            cameraOverlay.style.display = 'none';

            // 开始定期检测
            startPeriodicDetection();
        } catch (err) {
            alert('无法访问摄像头: ' + err.message);
        }
    });

    function startPeriodicDetection() {
        setInterval(async function() {
            if (isProcessing) return;
            isProcessing = true;

            // 捕获帧
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            // 转换为blob并发送
            canvas.toBlob(async function(blob) {
                const formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');

                try {
                    const response = await fetch('/api/detect_camera', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (data.success) {
                        updateCameraResults(data);
                    }
                } catch (e) {
                    console.error('检测失败:', e);
                }

                isProcessing = false;
            }, 'image/jpeg', 0.8);
        }, 2000); // 每2秒检测一次
    }

    function updateCameraResults(data) {
        let html = `检测到 ${data.vehicle_count} 辆车`;
        if (data.vehicles && data.vehicles.length > 0) {
            html += '<br>';
            data.vehicles.forEach(v => {
                html += `#${v.id}: ${v.type} | ${v.color}`;
                if (v.plates.length > 0) {
                    html += ` | 车牌: ${v.plates.join(', ')}`;
                }
                html += '<br>';
            });
        }
        cameraResults.innerHTML = html;
    }
});
