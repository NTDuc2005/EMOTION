import torch
from model_cnn import EmotionCNN  # hoặc ModalEncoder, tùy bạn dùng

# Load model
model = EmotionCNN(num_classes=7)

# Load state_dict từ file .pth
state_dict = torch.load("models/emotion_model.pth", map_location="cpu")

# In thông tin conv1 weights
for k, v in state_dict.items():
    if "conv1.0.weight" in k:
        print(f"{k} shape:", v.shape)
        if v.shape[1] == 3:
            print("→ Model train ảnh RGB (3 kênh)")
        elif v.shape[1] == 1:
            print("→ Model train ảnh grayscale (1 kênh)")
        else:
            print("→ Số kênh input khác:", v.shape[1])
