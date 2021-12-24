import torch
import torch.nn as nn
from torchsummary import summary


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, activation=True, **kwargs):
        super(BasicConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down
            else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            BasicConvBlock(channels, channels, kernel_size=3, padding=1),
            BasicConvBlock(channels, channels, activation=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlocks(nn.Module):
    def __init__(self, num_features):
        super(DownBlocks, self).__init__()
        self.down_blocks = nn.ModuleList(
            [
                BasicConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                BasicConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1)
            ]
        )
    
    def forward(self, x):
        for block in self.down_blocks:
            x = block(x)
        
        return x


class UpBlocks(nn.Module):
    def __init__(self, num_features):
        super(UpBlocks, self).__init__()
        self.up_blocks = nn.ModuleList(
            [
                BasicConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                BasicConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )

    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)
        
        return x


    
class Generator(nn.Module):
    def __init__(self, in_channels, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = DownBlocks(num_features)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = UpBlocks(num_features)

        self.final_block = nn.Conv2d(num_features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    
    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        x = self.final_block(x)
        return x


if __name__ == "__main__":
    in_channels = 3
    img_size = 256
    x = torch.rand((1, in_channels, img_size, img_size))
    gen = Generator(in_channels)
    print(gen(x).shape)