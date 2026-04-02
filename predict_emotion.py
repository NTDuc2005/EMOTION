from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import EMOTION_IMG_SIZE, EMOTION_LABELS, EMOTION_MODEL_PATH, YOLO_FACE_WEIGHTS
from model_cnn import EmotionCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_model = YOLO(YOLO_FACE_WEIGHTS)
emotion_model = EmotionCNN(num_class=len(EMOTION_LABELS)).to(DEVICE)

state_dict = torch.load(EMOTION_MODEL_PATH, map_location=DEVICE)
model_dict = emotion_model.state_dict()

fixed_dict = {}
for k, v in state_dict.items():
    # Backward compatibility for old key format fc3.0.*
    new_k = k.replace("fc3.0.", "fc3.") if k.startswith("fc3.0.") else k
    fixed_dict[new_k] = v

pretrained_dict = {k: v for k, v in fixed_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
emotion_model.load_state_dict(model_dict)
emotion_model.eval()


def _prepare_tensor(face_bgr: np.ndarray) -> torch.Tensor:
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, EMOTION_IMG_SIZE)
    face_gray = face_gray.astype("float32") / 255.0
    face_gray = np.expand_dims(face_gray, axis=0)
    face_gray = np.expand_dims(face_gray, axis=0)
    return torch.tensor(face_gray, dtype=torch.float32, device=DEVICE)


def predict_emotion(frame: np.ndarray, padding: int = 10) -> List[Dict]:
    results = face_model(frame, conf=0.5)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return []

    h, w, _ = frame.shape
    output: List[Dict] = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        with torch.no_grad():
            tensor = _prepare_tensor(face)
            logits = emotion_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        output.append(
            {
                "bbox": (x1, y1, x2, y2),
                "emotion": EMOTION_LABELS[idx],
                "confidence": float(probs[idx]),
            }
        )

    return output


def draw_emotion_results(frame: np.ndarray, emotion_results: List[Dict]) -> np.ndarray:
    for item in emotion_results:
        x1, y1, x2, y2 = item["bbox"]
        emotion = item["emotion"]
        conf = item["confidence"]
        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{emotion} ({conf * 100:.1f}%)",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = predict_emotion(frame)
        draw_emotion_results(frame, emotions)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
