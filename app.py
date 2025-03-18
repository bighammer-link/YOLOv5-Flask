import time

from flask import Flask, request, render_template, Response, redirect, flash
import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box

app = Flask(__name__)
app.secret_key = 'your_random_secret_key_here'

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# YOLOv5 模型加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('yolov5s.pt', map_location=device)
model.eval()
if device.type == 'cuda':
    model.half()
names = model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

imgsz = 160  # 进一步降低分辨率，减少推理时间
conf_thres = 0.25
iou_thres = 0.45

cap = None
video_path = None  # 保存上传的视频路径


# 公共检测函数：处理单帧
def detect_frame(frame):
    img = frame.copy()
    with torch.no_grad():
        stride = int(model.stride.max())
        imgsz_check = check_img_size(imgsz, s=stride)
        img = letterbox(img, new_shape=imgsz_check)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.half() else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=2)
    return frame


# 图片检测
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('未上传文件', 'error')
            return render_template('index.html')
        file = request.files['file']
        if not file:
            flash('请选择图片再提交', 'error')
            return render_template('index.html')

        # 读取图片
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        showimg = img.copy()

        # 检测
        showimg = detect_frame(showimg)

        # 保存结果
        img_file = 'static/img.jpg'
        cv2.imwrite(img_file, showimg)
        flash('图片检测完成', 'success')
        return render_template('index.html', detection_done=True)

    return render_template('index.html', detection_done=False)


# 上传视频
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path
    if 'video' not in request.files:
        flash('未上传视频', 'error')
        return redirect('/')
    video_file = request.files['video']
    if not video_file:
        flash('请选择视频文件再提交', 'error')
        return redirect('/')

    # 保存上传的视频
    video_path = 'static/uploaded_video.mp4'
    video_file.save(video_path)

    # 检查视频文件是否有效
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        flash('无法打开视频文件', 'error')
        logger.error(f"Failed to open video file: {video_path}")
        return redirect('/')
    cap.release()

    flash('视频上传成功，请点击“开始检测”', 'success')
    logger.info(f"Video uploaded successfully: {video_path}")
    return redirect('/')


# 原始视频流
@app.route('/original_video_feed')
def original_video_feed():
    global video_path
    if video_path is None:
        logger.error("No video uploaded")
        return "No video uploaded", 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return "Unable to open video", 500

    logger.info("Starting original video feed")
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始帧率
    frame_time = 1 / fps if fps > 0 else 0.03  # 防止除以 0，默认 30 FPS
    def generate():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("Original video feed ended")
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(frame_time)
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 获取视频第一帧
@app.route('/video_first_frame')
def video_first_frame():
    global video_path
    if video_path is None:
        logger.error("No video uploaded")
        return "No video uploaded", 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return "Unable to open video", 500

    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read first frame")
        cap.release()
        return "Failed to read first frame", 500

    ret, buffer = cv2.imencode('.jpg', frame)
    cap.release()
    if not ret:
        logger.error("Failed to encode first frame")
        return "Failed to encode first frame", 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')


# 检测后的视频流
@app.route('/processed_video_feed')
def processed_video_feed():
    global video_path
    if video_path is None:
        logger.error("No video uploaded")
        return "No video uploaded", 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Unable to open video: {video_path}")
        return "Unable to open video", 500
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始帧率
    frame_time = 1 / fps if fps > 0 else 0.03  # 防止除以 0，默认 30 FPS
    logger.info("Starting processed video feed")

    def generate():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("Processed video feed ended")
                break
            # 检测
            frame = detect_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(frame_time)
        cap.release()
        flash('视频检测完成', 'success')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 摄像头检测
def gen_frames():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 检测
        frame = detect_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_video')
def stop_video():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        logger.info("摄像头已释放")
    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
