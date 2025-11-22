import cv2
import numpy as np
import torch
from ultralytics import YOLO
from model_cnn import EmotionCNN  # model grayscale

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_PATH = "models/emotion_model.pth"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_model = YOLO("models/yolov8n-face-lindevs.pt")

#Emotion CNN
emotion_model = EmotionCNN(num_class=7).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model_dict = emotion_model.state_dict()

fixed_dict = {}
for k, v in state_dict.items():
    if k.startswith("fc3.0."):
        new_k = k.replace("fc3.0.", "fc3.")
    else:
        new_k = k
    fixed_dict[new_k] = v

pretrained_dict = {k: v for k, v in fixed_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
emotion_model.load_state_dict(model_dict)
emotion_model.eval()

# Hàm nhận diện nhiều khuôn mặt
def predict_emotion(frame, padding=10):
    results = face_model(frame, conf=0.5)
    r = results[0]

    h, w, _ = frame.shape
    emotions = []

    if not r.boxes or len(r.boxes) == 0:
        emotions.append(("Neutral", frame, 1.0))
        return emotions

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1-padding), max(0, y1-padding)
        x2, y2 = min(w, x2+padding), min(h, y2+padding)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            emotions.append(("Neutral", frame, 1.0))
            continue

        # Chuyển sang grayscale 48x48
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (48,48))
        face_gray = face_gray.astype("float32") / 255.0
        face_gray = np.expand_dims(face_gray, axis=0)
        face_gray = np.expand_dims(face_gray, axis=0)
        face_tensor = torch.tensor(face_gray, device=DEVICE)

        # Predict
        with torch.no_grad():
            outputs = emotion_model(face_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        idx = np.argmax(probs)
        emotion = emotion_labels[idx]
        confidence = float(probs[idx])

        # Vẽ bounding box + nhãn
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        emotions.append((emotion, frame, confidence))

    return emotions


#webcam
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = predict_emotion(frame)
        # frame đã được vẽ bounding box tất cả khuôn mặt
        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
