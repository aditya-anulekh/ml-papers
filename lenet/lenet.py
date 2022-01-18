import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    x = torch.rand([1, 1, 32, 32], device=device)
    net = LeNet(num_classes=10)
    net = net.to(device)
    net.training = False
    y = net(x)
    print(y.shape)