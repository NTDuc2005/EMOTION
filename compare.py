import argparse
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import FACE_CLASSES_PATH, FACE_IMG_SIZE, FACE_MODEL_PATH, YOLO_FACE_WEIGHTS
from face_model import FaceIdentityCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR = YOLO(YOLO_FACE_WEIGHTS)


def _crop_first_face(img):
    result = DETECTOR(img, conf=0.5)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0].tolist())
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _to_tensor(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_IMG_SIZE)
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=(0, 1))
    return torch.tensor(gray, dtype=torch.float32, device=DEVICE)


def _load_model():
    if not os.path.exists(FACE_MODEL_PATH) or not os.path.exists(FACE_CLASSES_PATH):
        raise FileNotFoundError("Chua co face model hoac classes")

    with open(FACE_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = FaceIdentityCNN(num_class=len(classes)).to(DEVICE)
    state = torch.load(FACE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _embedding(model, face):
    with torch.no_grad():
        emb = model.forward_features(_to_tensor(face))
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.squeeze(0)


def compare_two_images(img_a_path: str, img_b_path: str, threshold: float = 0.75):
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    if img_a is None or img_b is None:
        raise FileNotFoundError("Khong doc duoc mot trong hai anh")

    face_a = _crop_first_face(img_a)
    face_b = _crop_first_face(img_b)
    if face_a is None or face_b is None:
        raise ValueError("Khong tim thay khuon mat trong mot trong hai anh")

    model = _load_model()

    emb_a = _embedding(model, face_a)
    emb_b = _embedding(model, face_b)

    similarity = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
    return similarity, similarity >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-a", required=True)
    parser.add_argument("--img-b", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    sim, same = compare_two_images(args.img_a, args.img_b, args.threshold)
    print(f"cosine_similarity={sim:.4f}")
    print("same_person=yes" if same else "same_person=no")


if __name__ == "__main__":
    main()
