import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import FACE_IMG_SIZE, FACE_REGISTRY_PATH


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def face_to_embedding(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_IMG_SIZE)
    gray = cv2.equalizeHist(gray)

    # Low-dim intensity descriptor
    small = cv2.resize(gray, (16, 16)).astype("float32").flatten() / 255.0

    # Texture descriptor
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).astype("float32").flatten()
    hist = hist / max(float(hist.sum()), 1.0)

    emb = np.concatenate([small, hist], axis=0)
    return _l2_normalize(emb)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_registry(path: str = FACE_REGISTRY_PATH) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_registry(registry: Dict[str, Dict], path: str = FACE_REGISTRY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _first_face_crop(image: np.ndarray, detector, padding: int = 10) -> Optional[np.ndarray]:
    result = detector(image, conf=0.5)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    h, w, _ = image.shape
    x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0].tolist())
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def enroll_person(person_name: str, image_paths: List[str], detector, registry_path: str = FACE_REGISTRY_PATH) -> Tuple[int, str]:
    person_name = person_name.strip().lower()
    if not person_name:
        raise ValueError("Ten nguoi khong hop le.")

    embeddings = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        face = _first_face_crop(img, detector)
        if face is None:
            continue
        embeddings.append(face_to_embedding(face))

    if not embeddings:
        raise ValueError(f"Khong tim thay khuon mat hop le cho {person_name}.")

    centroid = _l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

    registry = load_registry(registry_path)
    registry[person_name] = {
        "embedding": centroid.tolist(),
        "samples": int(len(embeddings)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_registry(registry, registry_path)
    return len(embeddings), person_name


def identify_face(face_bgr: np.ndarray, registry: Dict[str, Dict], threshold: float) -> Tuple[str, float, bool]:
    if not registry:
        return "unknown", 0.0, False

    emb = face_to_embedding(face_bgr)

    best_name = "unknown"
    best_score = -1.0
    for name, item in registry.items():
        ref = np.asarray(item.get("embedding", []), dtype="float32")
        if ref.size == 0:
            continue
        score = cosine_similarity(emb, ref)
        if score > best_score:
            best_score = score
            best_name = name

    verified = best_score >= threshold
    return (best_name if verified else "unknown"), max(best_score, 0.0), verified
