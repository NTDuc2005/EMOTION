import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

#Mô hình YOLOv8
face_model = YOLO("models/yolov8n-face-lindevs.pt")
emotion_model = load_model("models/emotion_model.h5")

#Danh sách nhãn cảm xúc
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#Hàm nhận diện cảm xúc
def predict_emotion(frame):
    results = face_model(frame, conf=0.5)
    r = results[0]

    #Không có khuôn mặt nào
    if not r.boxes or len(r.boxes) == 0:
        return None, frame, None

    emotion_label = None
    emotion_probs = None  #Lưu xác suất

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        #Lấy kích thước input của mô hình (cao, rộng, kênh)
        input_shape = emotion_model.input_shape[1:4]
        img_h, img_w, channels = input_shape

        #Resize khuôn mặt đúng kích thước
        face = cv2.resize(face, (img_w, img_h))

        #Chuyển ảnh sang đúng định dạng kênh 1 3
        if channels == 1:  #Mô hình dùng ảnh xám
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=-1)
        else:  #Mô hình dùng ảnh màu
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        #Chuẩn hóa
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = emotion_model.predict(face, verbose=0)
        emotion_probs = prediction[0]  # lưu mảng xác suất
        emotion_label = emotion_labels[np.argmax(prediction)]

        #Hiển thị kết quả
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return emotion_label, frame, emotion_probs
