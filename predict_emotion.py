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

    if not r.boxes or len(r.boxes) == 0:
        return "Neutral", frame, np.zeros(len(emotion_labels))  # luôn trả giá trị

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        img_h, img_w, channels = emotion_model.input_shape[1:4]
        face = cv2.resize(face, (img_w, img_h))

        if channels == 1:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=-1)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = emotion_model.predict(face, verbose=0)
        emotion_label = emotion_labels[np.argmax(prediction)]
        emotion_probs = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Trả ngay emotion đầu tiên detect được
        return emotion_label, frame, emotion_probs

    # Nếu không detect được face hợp lệ
    return "Neutral", frame, np.zeros(len(emotion_labels))

