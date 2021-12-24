import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=stride,
                      padding=1,
                      bias=True,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels,
                                out_channels=feature,
                                stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=1,
                                kernel_size=4,
                                padding=1,
                                padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = torch.sigmoid(self.model(x))
        return x


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using {device}")
#     x = torch.randn(size=(5, 3, 256, 256), device=device)
#     model = Discriminator(in_channels=3).to(device)
#     preds = model(x)
#     summary(model, input_size=(3, 256, 256))
#     print(preds.shape)

