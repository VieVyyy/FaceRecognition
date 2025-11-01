# Mô hình Face Recognition

## Tổng quan

Mô hình Face Recognition đơn giản sử dụng các pre-trained model

- Face detector: [Pre-trained model RetinaFace](https://github.com/biubug6/Pytorch_Retinaface.git)
- Embedder: [Pre-trained model MobileFace Net](https://github.com/foamliu/MobileFaceNet.git)
- Database: Faiss
- Web demo: Flask

## Cách chạy trên máy

1. Clone repos

```
git clone https://github.com/VieVyyy/FaceRecognition.git
```

2. Cài đặt môi trường

Phiên bản Python: `3.11.9`

```
cd FaceRecognition
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

3. Chạy Flask

```
python main.py
```

## Các tính năng
### Nhận diện khuôn mặt

Click nút `Start stream` để khởi động camera bắt đầu nhận diện.
Khuôn mặt detect được sẽ được vẽ bounding box.

- Khuôn mặt đã có trong data base -> gán label tương ứng.
- Khuôn mặt chưa có trong data base (không nhận diện được) -> gán label "unknown".

Click nút `Stop stream` để dừng.

### Thêm khuôn mặt mới

Click nút `Add new face` để vào trang thêm khuôn mặt với vào data base.

1. Nhập tên (label).
2. Click nút `Capture & Add Face` để tiến hành thêm.

Sau khi hiển thị thêm thành công có thể quay về trang Nhận diện để nhận diện khuôn mặt mới.
