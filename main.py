import os
import cv2
import torch
import numpy as np
from src.face_recognizer import FaceRecognizer
from src.faiss_index import FaceIndex
from utils.image_utils import *

from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
from threading import Lock

# Khai báo ứng dụng Flask và SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_INDEX_PATH = "face_data/emb_face/face_index.bin"
THRESHOLD = 1.3  # Ngưỡng khoảng cách để xác định Unknown (càng nhỏ càng chặt)
is_streaming = False # Biến trạng thái luồng video

detector_model = "models/RetinaFaceNet.onnx"
embedder_model = "models/MobileFaceNet.onnx"

# Khởi tạo nhận diện và index
recognizer = FaceRecognizer(detector_model, embedder_model, device=DEVICE)
faiss_index = FaceIndex(dim=128, index_path=FACE_INDEX_PATH)
model_lock = Lock()

org_img_dir = "face_data/org_face/"
if not os.path.exists(FACE_INDEX_PATH):
    print("Tạo FAISS index từ ảnh gốc...")
    faiss_index.create(org_img_dir, recognizer)

print("Face recognizer and FAISS index loaded.")

def encodeFrameToBase64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

@socketio.on('start_stream')
def handleStartStream():
    global is_streaming
    if not is_streaming:
        print("Client kết nối real-time face recognition.")
        is_streaming = True
        socketio.start_background_task(target=streamVideo)
    else:
        print("Stream đang chạy!")

@socketio.on('stop_stream')
def handleStopStream():
    global is_streaming
    if is_streaming:
        print("Client ngắt kết nối real-time face recognition.")
        is_streaming = False
    else:
        print("Stream chưa chạy!")

def streamVideo(threshold=1.1):
    global is_streaming 

    cap = cv2.VideoCapture(0)
    frame_id = 0
    if not cap.isOpened():
        print("Không thể mở camera!")
        is_streaming = False
        socketio.emit('stream_error', {'message': 'Không thể mở camera!'})
        return
    
    try:
        while is_streaming:
            success, frame = cap.read()
            if not success:
                print("Không đọc được khung hình!")
                break

            frame_id += 1
            results = []

            with model_lock:
                aligned_faces, bboxes, _, _ = recognizer.detectAndAlign(frame)

                for face, bbox in zip(aligned_faces, bboxes):
                    emb = recognizer.getEmbeddingFromAligned(face)
                    if emb is not None:
                        labels, distances = faiss_index.search(emb, top_k=1)
                        label = labels[0]
                        distance = distances[0]

                        if distance > threshold:
                            label = "Unknown"

                        results.append((label, distance, bbox))

            for label, distance, bbox in results:
                if label == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({distance:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            frame_base64 = encodeFrameToBase64(frame)
            socketio.emit('video_frame', {'image': frame_base64})
            socketio.sleep(0.03)  # Giới hạn tốc độ khung hình

            if frame_id % 15 == 0:
                del frame, aligned_faces, bboxes, results
                torch.cuda.empty_cache() # Giải phóng bộ nhớ GPU định kỳ

    finally:
        print("Dừng stream video.")
        cap.release()
        is_streaming = False
        socketio.emit('stream_stopped', {'message': 'Stream đã dừng.'})

def addFaceToIndex(image_array, label):
    i = 1
    save_path = os.path.join(org_img_dir, f"{label}.jpg")
    while os.path.exists(save_path):
        save_path = os.path.join(org_img_dir, f"{label}_{i}.jpg")
        i += 1

    aligned_faces = []
    emb = None

    with model_lock:
        print(f"Đang thêm khuôn mặt cho {label}...")
        aligned_faces, _, _, _ = recognizer.detectAndAlign(image_array)

        if len(aligned_faces) == 0:
            print(f"Không phát hiện khuôn mặt cho {label}.")
            return False, f"Không phát hiện khuôn mặt."
        
        emb = recognizer.getEmbeddingFromAligned(aligned_faces[0])
        if emb is None:
            print(f"Không thể trích xuất embedding cho {label}.")
            return False, f"Không thể trích xuất embedding."
        
        faiss_index.add(emb, label)
        faiss_index.saveIndex()

    cv2.imwrite(save_path, image_array)
    print(f"Đã thêm khuôn mặt cho {label} vào index và lưu ảnh tại {save_path}.")
    return True, f"Thêm thành công '{label}' vào index."


@socketio.on('add_face')
def handleAddFace(data):
    name = data.get('label')
    img_base64 = data.get('image')

    if not name or not img_base64:
        socketio.emit('add_face_response', {'status': 'error', 'message': 'Tên hoặc ảnh không hợp lệ.'})
        return
    
    try:
        # Giải mã ảnh từ base64
        img_data = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_array is None:
            socketio.emit('add_face_response', {'success': False, 'message': 'Ảnh không hợp lệ.'})
            return
        
        success, message = addFaceToIndex(img_array, name)

        if success:
            socketio.emit('add_face_response', {'success': success, 'message': message})
        else:
            socketio.emit('add_face_response', {'success': False, 'message': message})
    except Exception as e:
        message = f"Lỗi khi thêm khuôn mặt: {str(e)}"
        socketio.emit('add_face_response', {'success': False, 'message': message})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add')
def add_face():
    return render_template('add_face.html')

if __name__ == '__main__':
    socketio.run(app, debug=False)