import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()

        self.conv1 = self.make_block(1, 64)
        self.conv2 = self.make_block(64, 128)
        self.conv3 = self.make_block(128, 256)
        self.conv4 = self.make_block(256, 512)

        self.flatten_dim = 512 * 3 * 3

        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024, num_class)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


if __name__ == "__main__":
    model = EmotionCNN()
    dummy = torch.randn(1, 1, 48, 48)
    out = model(dummy)
    print("Output:", out.shape)
