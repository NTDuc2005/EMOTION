import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class FaceIdentityCNN(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)
        return self.classifier(x)


def build_transform(img_size: int, train: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
    ]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / max(total, 1)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None
    output_model = Path(args.output_model)
    output_classes = Path(args.output_classes)

    if not train_dir.exists():
        raise FileNotFoundError(f'Khong tim thay thu muc train: {train_dir}')

    train_ds = datasets.ImageFolder(str(train_dir), transform=build_transform(args.img_size, train=True))
    if len(train_ds.classes) < 1:
        raise ValueError('Khong co class nao trong train_dir')

    test_ds = None
    if test_dir and test_dir.exists():
        test_ds = datasets.ImageFolder(str(test_dir), transform=build_transform(args.img_size, train=False))
        if set(test_ds.classes) != set(train_ds.classes):
            raise ValueError('Class trong test_dir khong khop train_dir')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    model = FaceIdentityCNN(num_class=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_metric = 0.0
    best_state = model.state_dict()

    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            batch_total = labels.size(0)
            total += batch_total
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * batch_total

            pbar.set_postfix(
                loss=f'{loss_sum / max(total, 1):.4f}',
                acc=f'{correct / max(total, 1):.4f}',
            )

        train_acc = correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)

        metric_name = 'train_acc'
        metric_value = train_acc
        if test_loader is not None:
            metric_name = 'val_acc'
            metric_value = evaluate(model, test_loader, device)

        if metric_value >= best_metric:
            best_metric = metric_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f'Epoch {epoch + 1}: '
            f'train_loss={train_loss:.4f}, '
            f'train_acc={train_acc:.4f}, '
            f'{metric_name}={metric_value:.4f}'
        )

    model.load_state_dict(best_state)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_classes.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_model)
    with open(output_classes, 'w', encoding='utf-8') as f:
        json.dump(train_ds.classes, f, ensure_ascii=False, indent=2)

    print(f'Saved model: {output_model}')
    print(f'Saved classes: {output_classes}')
    print(f'Best metric: {best_metric:.4f}')
    print('Classes:', train_ds.classes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face identity model on Google Colab')
    parser.add_argument('--train-dir', required=True, help='Thu muc train: <train_dir>/<ten_nguoi>/*.jpg')
    parser.add_argument('--test-dir', default='', help='Thu muc test: <test_dir>/<ten_nguoi>/*.jpg')
    parser.add_argument('--output-model', required=True, help='Noi luu file model .pth')
    parser.add_argument('--output-classes', required=True, help='Noi luu file classes .json')
    parser.add_argument('--img-size', type=int, default=112)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
