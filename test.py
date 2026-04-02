import os

import torch
from torchvision import datasets

from config import EMOTION_LABELS, EMOTION_MODEL_PATH, FACE_CLASSES_PATH, FACE_MODEL_PATH, TRAIN_DIR
from face_model import FaceIdentityCNN
from model_cnn import EmotionCNN


def check_emotion_model():
    print("== Emotion Model ==")
    if not os.path.exists(EMOTION_MODEL_PATH):
        print("Chua co emotion model:", EMOTION_MODEL_PATH)
        return

    model = EmotionCNN(num_class=len(EMOTION_LABELS))
    state = torch.load(EMOTION_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print("Load emotion model OK")


def check_emotion_class_order():
    print("\n== Emotion Dataset Classes ==")
    ds = datasets.ImageFolder(TRAIN_DIR)
    print("classes:", ds.classes)
    if ds.classes != EMOTION_LABELS:
        print("CANH BAO: thu tu class dataset khac config.")
    else:
        print("Class order khop config.")


def check_face_model():
    print("\n== Face Model ==")
    print("face model exists:", os.path.exists(FACE_MODEL_PATH))
    print("face classes exists:", os.path.exists(FACE_CLASSES_PATH))
    if os.path.exists(FACE_CLASSES_PATH):
        import json

        with open(FACE_CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)
        print("so nguoi da hoc:", len(classes))
        model = FaceIdentityCNN(num_class=len(classes))
        state = torch.load(FACE_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("Load face model OK")


if __name__ == "__main__":
    check_emotion_model()
    check_emotion_class_order()
    check_face_model()
