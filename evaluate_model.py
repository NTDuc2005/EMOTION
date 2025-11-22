import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model_cnn import EmotionCNN

# --- Config ---
TEST_DIR = "data/test"
MODEL_PATH = "cnn_emotion_rgb.pth"
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Transform ---
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load model ---
model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = nn.CrossEntropyLoss()

# --- Evaluate ---
all_preds = []
all_probs = []
all_labels = []

running_loss = 0.0
total = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_loss = running_loss / total
test_acc = correct / total

print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

#Dự đoán xác suất cho từng ảnh
all_probs = np.concatenate(all_probs, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

#Hiển thị 5 ảnh ngẫu nhiên cùng xác suất
indices = np.random.choice(len(test_dataset), 5, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    img_np = np.transpose(img.numpy(), (1, 2, 0))
    true_label = EMOTION_LABELS[label]
    pred_label = EMOTION_LABELS[all_preds[idx]]
    probs = all_probs[idx]

    plt.subplot(2, 5, i+1)
    plt.imshow(img_np)
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

    plt.subplot(2, 5, i+6)
    plt.bar(EMOTION_LABELS, probs)
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    plt.title("Probabilities")

plt.tight_layout()
plt.show()

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Classification")
plt.show()
