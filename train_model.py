import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from config import EMOTION_IMG_SIZE, EMOTION_LABELS, EMOTION_MODEL_PATH, TEST_DIR, TRAIN_DIR
from model_cnn import EmotionCNN

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(train: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(EMOTION_IMG_SIZE),
    ]
    if train:
        ops.extend([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
        ])
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def run():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=build_transform(train=True))
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=build_transform(train=False))

    if len(train_dataset.classes) != len(EMOTION_LABELS):
        raise ValueError(
            f"So lop train ({len(train_dataset.classes)}) khac voi config ({len(EMOTION_LABELS)})."
        )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EmotionCNN(num_class=len(EMOTION_LABELS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_total, val_correct = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_wts)
    os.makedirs(os.path.dirname(EMOTION_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), EMOTION_MODEL_PATH)

    print(f"Saved model: {EMOTION_MODEL_PATH}")
    print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    run()
