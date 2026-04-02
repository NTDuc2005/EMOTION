import json
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    FACE_CLASSES_PATH,
    FACE_IMG_SIZE,
    FACE_MIN_CONFIDENCE,
    FACE_MODEL_PATH,
    YOLO_FACE_WEIGHTS,
)
from face_model import FaceIdentityCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_DETECTOR = YOLO(YOLO_FACE_WEIGHTS)


def _load_face_assets():
    if not os.path.exists(FACE_MODEL_PATH) or not os.path.exists(FACE_CLASSES_PATH):
        return None, []

    with open(FACE_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = FaceIdentityCNN(num_class=len(classes)).to(DEVICE)
    state = torch.load(FACE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, [str(x).lower() for x in classes]


FACE_MODEL, FACE_CLASSES = _load_face_assets()


def reload_face_assets():
    global FACE_MODEL, FACE_CLASSES
    FACE_MODEL, FACE_CLASSES = _load_face_assets()
    return FACE_MODEL is not None and bool(FACE_CLASSES)


def is_face_model_ready() -> bool:
    return FACE_MODEL is not None and bool(FACE_CLASSES)


def _preprocess(face_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_IMG_SIZE)
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=0)
    return torch.tensor(gray, dtype=torch.float32, device=DEVICE)


def predict_face_id(frame, threshold: float = FACE_MIN_CONFIDENCE, padding: int = 10) -> List[Dict]:
    outputs: List[Dict] = []

    result = FACE_DETECTOR(frame, conf=0.5)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return outputs

    h, w, _ = frame.shape

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        identity = "unknown"
        conf = 0.0
        verified = False

        if FACE_MODEL is not None and FACE_CLASSES:
            with torch.no_grad():
                x = _preprocess(crop)
                logits = FACE_MODEL(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                pred_name = FACE_CLASSES[idx]
                if conf >= threshold:
                    identity = pred_name
                    verified = True

        outputs.append(
            {
                "bbox": (x1, y1, x2, y2),
                "identity": identity,
                "confidence": conf,
                "verified": verified,
            }
        )

    return outputs


def draw_face_results(frame, face_results: List[Dict]):
    for item in face_results:
        x1, y1, x2, y2 = item["bbox"]
        name = item["identity"]
        conf = item["confidence"]
        verified = item["verified"]

        color = (0, 180, 0) if verified else (0, 120, 255)
        label = f"ID: {name} ({conf * 100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = predict_face_id(frame)
        draw_face_results(frame, faces)

        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
