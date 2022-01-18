import json
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, **kwargs)
        self.bn = nn.BatchNorm2d(kwargs["out_channels"])
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class YOLO(nn.Module):
    def __init__(self, architecture, S, B, C, in_channels=3):
        super(YOLO, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fc = self._create_fc(S, B, C)

    def forward(self, x):
        x = self.darknet(x)
        x = self.fc(x)
        return x

    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        for layer in self.architecture["architecture"]:
            if layer["type"] == "C":
                layers+= [
                    ConvBlock(in_channels, **layer["kwargs"])
                ]
                in_channels = layer["kwargs"]["out_channels"]
            elif layer["type"] == "M":
                layers += [
                    nn.MaxPool2d(**layer["kwargs"])
                ]
            
        return nn.Sequential(*layers)

    def _create_fc(self, S, B, C):
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, (S*S*(B*5+C)))
        )

        return fc

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    with open("yolo/architecture.json", "r") as file:
        arch = json.load(file)

    x = torch.rand([1,3,448,448], device=device)
    model = YOLO(arch, 7, 2, 20, 3).to(device)
    y = model(x)
    print(y.size())