import argparse
import copy
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from config import FACE_CLASSES_PATH, FACE_IMG_SIZE, FACE_MODEL_PATH, FACE_TEST_DIR, FACE_TRAIN_DIR
from face_model import FaceIdentityCNN

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(train: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(FACE_IMG_SIZE),
    ]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=FACE_TRAIN_DIR, help="Thu muc train (co the dat tren Google Drive)")
    parser.add_argument("--test-dir", default=FACE_TEST_DIR, help="Thu muc test (optional)")
    parser.add_argument("--output-model", default=FACE_MODEL_PATH, help="Duong dan luu model .pth")
    parser.add_argument("--output-classes", default=FACE_CLASSES_PATH, help="Duong dan luu classes .json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    return parser.parse_args()

def run(train_dir, test_dir, output_model, output_classes, epochs, batch_size, lr):
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Khong tim thay du lieu train: {train_dir}. "
            "Tao thu muc <train_dir>/<ten_nguoi>/*.jpg"
        )

    train_ds = datasets.ImageFolder(train_dir, transform=build_transform(train=True))
    if len(train_ds.classes) < 1:
        raise ValueError("Khong co class nao trong train_dir")

    test_ds = None
    if test_dir and os.path.isdir(test_dir):
        test_ds = datasets.ImageFolder(test_dir, transform=build_transform(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds and len(test_ds) > 0 else None

    model = FaceIdentityCNN(num_class=len(train_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_metric = 0.0
    best_w = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)

        metric_name = "train"
        metric_value = train_acc

        if test_loader is not None:
            model.eval()
            t_total, t_correct = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    t_total += labels.size(0)
                    t_correct += (preds == labels).sum().item()
            metric_name = "val"
            metric_value = t_correct / max(t_total, 1)

        if metric_value > best_metric:
            best_metric = metric_value
            best_w = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"{metric_name}_acc={metric_value:.4f}"
        )

    model.load_state_dict(best_w)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    os.makedirs(os.path.dirname(output_classes), exist_ok=True)

    torch.save(model.state_dict(), output_model)

    with open(output_classes, "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)

    print(f"Saved face model: {output_model}")
    print(f"Saved face classes: {output_classes}")
    print(f"Best {metric_name} acc: {best_metric:.4f}")


if __name__ == "__main__":
    args = parse_args()
    run(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_model=args.output_model,
        output_classes=args.output_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
