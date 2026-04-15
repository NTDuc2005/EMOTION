import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    FACE_CLASSES_PATH,
    FACE_IMG_SIZE,
    FACE_MIN_CONFIDENCE,
    FACE_MODEL_PATH,
    ROOT,
    YOLO_FACE_WEIGHTS,
)
from face_model import FaceIdentityCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_DETECTOR = YOLO(YOLO_FACE_WEIGHTS)

# =========================
# Cau hinh nhan dien mat
# =========================
MIN_FACE_SIZE = 72
FACE_PADDING = 12
MIN_DETECT_CONFIDENCE = 0.60

# Giam nhe threshold de de nhan hon khi test
MIN_IDENTITY_CONFIDENCE = max(float(FACE_MIN_CONFIDENCE), 0.45)
MIN_IDENTITY_MARGIN = 0.08


def detect_face_boxes(
    frame: np.ndarray,
    detect_conf: float = MIN_DETECT_CONFIDENCE,
    padding: int = FACE_PADDING,
    min_face_size: int = MIN_FACE_SIZE,
) -> List[Dict[str, object]]:
    detections: List[Dict[str, object]] = []
    result = FACE_DETECTOR(frame, conf=detect_conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    h, w, _ = frame.shape
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = (
            float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
        )

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        width = x2 - x1
        height = y2 - y1
        if width < min_face_size or height < min_face_size:
            continue

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "detect_confidence": confidence,
                "area": width * height,
            }
        )

    detections.sort(
        key=lambda item: (
            -float(item.get("detect_confidence", 0.0)),
            -int(item.get("area", 0)),
        )
    )
    return detections


def _resolve_face_asset_paths() -> Tuple[str, str]:
    model_path = FACE_MODEL_PATH
    classes_path = FACE_CLASSES_PATH

    legacy_model_path = os.path.join(ROOT, "face_model", "face_model.pth")
    legacy_classes_path = os.path.join(ROOT, "face_model", "face_classes.json")

    if not os.path.exists(model_path) and os.path.exists(legacy_model_path):
        model_path = legacy_model_path
    if not os.path.exists(classes_path) and os.path.exists(legacy_classes_path):
        classes_path = legacy_classes_path

    return model_path, classes_path


def _load_face_assets() -> Tuple[torch.nn.Module | None, List[str]]:
    model_path, classes_path = _resolve_face_asset_paths()
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print(f"[FACE] Missing model/classes: {model_path} | {classes_path}")
        return None, []

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = FaceIdentityCNN(num_class=len(classes)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[FACE] model_path={model_path}")
    print(f"[FACE] classes_path={classes_path}")
    print(f"[FACE] num_classes={len(classes)}")
    print(f"[FACE] classes={classes}")
    print(f"[FACE] missing_keys={missing}")
    print(f"[FACE] unexpected_keys={unexpected}")

    model.eval()
    return model, [str(x) for x in classes]


FACE_MODEL, FACE_CLASSES = _load_face_assets()


def reload_face_assets() -> bool:
    global FACE_MODEL, FACE_CLASSES
    FACE_MODEL, FACE_CLASSES = _load_face_assets()
    return FACE_MODEL is not None and bool(FACE_CLASSES)


def is_face_model_ready() -> bool:
    return FACE_MODEL is not None and bool(FACE_CLASSES)


def _preprocess(face_bgr: np.ndarray, flip: bool = False) -> torch.Tensor:
    # Khop voi pipeline train:
    # Grayscale(1) -> Resize -> ToTensor()
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_IMG_SIZE)

    if flip:
        gray = cv2.flip(gray, 1)

    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=0)

    return torch.tensor(gray, dtype=torch.float32, device=DEVICE)


def _predict_probs(face_bgr: np.ndarray) -> np.ndarray:
    if FACE_MODEL is None or not FACE_CLASSES:
        return np.empty((0,), dtype=np.float32)

    with torch.no_grad():
        original = FACE_MODEL(_preprocess(face_bgr, flip=False))
        mirrored = FACE_MODEL(_preprocess(face_bgr, flip=True))
        logits = (original + mirrored) / 2.0
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs.astype(np.float32)


def _compute_margin(probs: np.ndarray) -> float:
    if probs.size < 2:
        return 0.0
    top2 = np.partition(probs, -2)[-2:]
    top1 = float(np.max(top2))
    second = float(np.min(top2))
    return top1 - second


def predict_face_id(
    frame: np.ndarray,
    threshold: float = MIN_IDENTITY_CONFIDENCE,
    detect_conf: float = MIN_DETECT_CONFIDENCE,
    padding: int = FACE_PADDING,
    detections: Optional[List[Dict[str, object]]] = None,
) -> List[Dict]:
    outputs: List[Dict] = []

    face_boxes = (
        detections
        if detections is not None
        else detect_face_boxes(
            frame,
            detect_conf=detect_conf,
            padding=padding,
            min_face_size=MIN_FACE_SIZE,
        )
    )

    if not face_boxes:
        return outputs

    class_count = len(FACE_CLASSES)

    for detection in face_boxes:
        x1, y1, x2, y2 = detection["bbox"]
        width = x2 - x1
        height = y2 - y1
        detect_score = float(detection.get("detect_confidence", 0.0))

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        identity = "unknown"
        display_identity = "unknown"
        raw_identity = "unknown"
        conf = 0.0
        verified = False
        margin = 0.0
        reason = "unknown"
        raw_probs: List[float] = []

        probs = _predict_probs(crop)
        if probs.size == 0 or class_count == 0:
            reason = "no_model_or_no_probs"

        elif class_count == 1:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            pred_name = FACE_CLASSES[idx]
            raw_identity = pred_name
            raw_probs = probs.tolist()
            identity = "unknown"
            display_identity = "unknown"
            verified = False
            margin = 0.0
            reason = "single_class_model"

        else:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            pred_name = FACE_CLASSES[idx]
            raw_identity = pred_name
            raw_probs = probs.tolist()
            margin = _compute_margin(probs)

            enough_detect_conf = detect_score >= detect_conf
            enough_identity_conf = conf >= threshold
            enough_margin = margin >= MIN_IDENTITY_MARGIN
            enough_face_size = min(width, height) >= MIN_FACE_SIZE

            print("=" * 60)
            print(f"[FACE] pred_name={pred_name}")
            print(f"[FACE] conf={conf:.4f}")
            print(f"[FACE] margin={margin:.4f}")
            print(f"[FACE] detect_score={detect_score:.4f}")
            print(f"[FACE] bbox={(x1, y1, x2, y2)}")
            top_k = min(5, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            print("[FACE] top_probs=", [(FACE_CLASSES[i], float(probs[i])) for i in top_indices])

            if (
                enough_detect_conf
                and enough_identity_conf
                and enough_margin
                and enough_face_size
            ):
                identity = pred_name
                display_identity = pred_name
                verified = True
                reason = "verified"
            else:
                identity = "unknown"
                display_identity = "unknown"
                verified = False
                parts = []
                if not enough_detect_conf:
                    parts.append("low_detect_conf")
                if not enough_identity_conf:
                    parts.append("low_identity_conf")
                if not enough_margin:
                    parts.append("low_margin")
                if not enough_face_size:
                    parts.append("small_face")
                reason = ",".join(parts) if parts else "unknown"

            print(f"[FACE] verified={verified} reason={reason}")

        outputs.append(
            {
                "bbox": (x1, y1, x2, y2),
                "identity": identity,
                "display_identity": display_identity,
                "raw_identity": raw_identity,
                "confidence": conf,
                "verified": verified,
                "margin": margin,
                "detect_confidence": detect_score,
                "reason": reason,
                "raw_probs": raw_probs,
            }
        )

    return outputs


def draw_face_results(frame: np.ndarray, face_results: List[Dict]) -> np.ndarray:
    for item in face_results:
        x1, y1, x2, y2 = item["bbox"]

        if item.get("verified"):
            predicted_name = item.get("identity", "unknown")
        else:
            predicted_name = "unknown"

        conf = float(item.get("confidence", 0.0))
        verified = bool(item.get("verified", False))
        margin = float(item.get("margin", 0.0))
        reason = str(item.get("reason", ""))

        color = (0, 180, 0) if verified else (0, 120, 255)
        label = f"ID: {predicted_name} ({conf * 100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

        if not verified:
            debug_text = f"unknown | m={margin:.2f} {reason}".strip()
            cv2.putText(
                frame,
                debug_text,
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
        ok, frame = cap.read()
        if not ok:
            break

        faces = predict_face_id(frame)
        frame = draw_face_results(frame, faces)

        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()