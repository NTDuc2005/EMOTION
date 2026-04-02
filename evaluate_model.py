import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    EMOTION_IMG_SIZE,
    EMOTION_LABELS,
    EMOTION_MODEL_PATH,
    FACE_CLASSES_PATH,
    FACE_IMG_SIZE,
    FACE_MODEL_PATH,
    FACE_TEST_DIR,
    TEST_DIR,
)
from face_model import FaceIdentityCNN
from model_cnn import EmotionCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / max(total, 1)


def evaluate_emotion():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(EMOTION_IMG_SIZE),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = EmotionCNN(num_class=len(EMOTION_LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=DEVICE))
    acc = evaluate(model, loader)
    print(f"Emotion test accuracy: {acc * 100:.2f}%")


def evaluate_face():
    if not os.path.isdir(FACE_TEST_DIR):
        print("Bo qua face evaluate: chua co FACE_TEST_DIR")
        return
    if not os.path.exists(FACE_MODEL_PATH) or not os.path.exists(FACE_CLASSES_PATH):
        print("Bo qua face evaluate: chua co face model/classes")
        return

    import json

    with open(FACE_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(FACE_IMG_SIZE),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(FACE_TEST_DIR, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FaceIdentityCNN(num_class=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=DEVICE), strict=False)
    acc = evaluate(model, loader)
    print(f"Face test accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    evaluate_emotion()
    evaluate_face()
