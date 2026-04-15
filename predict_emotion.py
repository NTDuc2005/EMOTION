from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from config import EMOTION_IMG_SIZE, EMOTION_LABELS, EMOTION_MODEL_PATH
from model_cnn import EmotionCNN
from predict_face import MIN_FACE_SIZE as FACE_MIN_SIZE_FOR_ID
from predict_face import detect_face_boxes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Siết chặt hơn để tránh nhảy bừa sang fear
EMOTION_MIN_MARGIN = 0.12
MIN_FACE_SIZE = 56
MIN_DETECT_CONFIDENCE = 0.60
EMOTION_MIN_CONFIDENCE = 0.55

EMOTION_MODEL = EmotionCNN(num_class=len(EMOTION_LABELS)).to(DEVICE)

state_dict = torch.load(EMOTION_MODEL_PATH, map_location=DEVICE)
model_dict = EMOTION_MODEL.state_dict()

fixed_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("fc3.0.", "fc3.") if k.startswith("fc3.0.") else k
    fixed_dict[new_k] = v

pretrained_dict = {
    k: v for k, v in fixed_dict.items()
    if k in model_dict and v.size() == model_dict[k].size()
}
model_dict.update(pretrained_dict)
EMOTION_MODEL.load_state_dict(model_dict)
EMOTION_MODEL.eval()


def _normalize_gray(face_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_gray = clahe.apply(face_gray)
    face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
    return face_gray


def _prepare_tensor(face_bgr: np.ndarray, flip: bool = False) -> torch.Tensor:
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, EMOTION_IMG_SIZE)
    face_gray = _normalize_gray(face_gray)

    if flip:
        face_gray = cv2.flip(face_gray, 1)

    face_gray = face_gray.astype(np.float32) / 255.0
    face_gray = (face_gray - 0.5) / 0.5
    face_gray = np.expand_dims(face_gray, axis=0)
    face_gray = np.expand_dims(face_gray, axis=0)

    return torch.tensor(face_gray, dtype=torch.float32, device=DEVICE)


def _predict_probs(face_bgr: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits_a = EMOTION_MODEL(_prepare_tensor(face_bgr, flip=False))
        logits_b = EMOTION_MODEL(_prepare_tensor(face_bgr, flip=True))
        probs = torch.softmax((logits_a + logits_b) / 2.0, dim=1).cpu().numpy()[0]
    return probs.astype(np.float32)


def predict_emotion(
    frame: np.ndarray,
    detect_conf: float = MIN_DETECT_CONFIDENCE,
    padding: int = 10,
    uncertain_threshold: float = EMOTION_MIN_CONFIDENCE,
    detections: Optional[List[Dict[str, object]]] = None,
) -> List[Dict]:
    face_boxes = detections if detections is not None else detect_face_boxes(
        frame,
        detect_conf=detect_conf,
        padding=padding,
        min_face_size=max(MIN_FACE_SIZE, FACE_MIN_SIZE_FOR_ID),
    )
    if not face_boxes:
        return []

    output: List[Dict] = []

    for detection in face_boxes:
        x1, y1, x2, y2 = detection["bbox"]
        detect_score = float(detection.get("detect_confidence", 0.0))

        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        probs = _predict_probs(face)
        if probs.size == 0:
            continue

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        predicted_emotion = EMOTION_LABELS[idx]

        if probs.size > 1:
            top2 = np.partition(probs, -2)[-2:]
            margin = float(top2[-1] - top2[-2])
        else:
            margin = conf

        reliable = (
            detect_score >= detect_conf
            and conf >= uncertain_threshold
            and margin >= EMOTION_MIN_MARGIN
        )

        emotion = predicted_emotion if reliable else "uncertain"

        output.append(
            {
                "bbox": (x1, y1, x2, y2),
                "emotion": emotion,                       # logic ky thuat
                "display_emotion": predicted_emotion,    # de hien thi tren app
                "raw_emotion": predicted_emotion,        # de debug
                "confidence": conf,
                "margin": margin,
                "reliable": reliable,
                "raw_probs": probs.tolist(),
                "detect_confidence": detect_score,
            }
        )

    return output


def draw_emotion_results(frame: np.ndarray, emotion_results: List[Dict]) -> np.ndarray:
    for item in emotion_results:
        x1, y1, x2, y2 = item["bbox"]
        shown_emotion = (
            item.get("display_emotion")
            or item.get("raw_emotion")
            or item.get("emotion")
            or "no-face"
        )
        conf = float(item["confidence"])
        reliable = bool(item.get("reliable", False))
        margin = float(item.get("margin", 0.0))

        color = (0, 180, 0) if reliable else (0, 165, 255)
        text = f"{shown_emotion} ({conf * 100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if not reliable:
            cv2.putText(
                frame,
                f"uncertain | m={margin:.2f}",
                (x1, min(frame.shape[0] - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1,
            )

    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotions = predict_emotion(frame)
        frame = draw_emotion_results(frame, emotions)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()