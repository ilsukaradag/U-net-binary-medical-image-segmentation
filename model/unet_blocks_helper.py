import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.encoder1 = DownBlock(in_channels, 64)
        self.encoder2 = DownBlock(64, 128)
        self.encoder3 = DownBlock(128, 256)
        self.encoder4 = DownBlock(256, 512)

    def forward(self, x):
        x1, x = self.encoder1(x)
        x2, x = self.encoder2(x)
        x3, x = self.encoder3(x)
        x4, x = self.encoder4(x)
        
        return x1, x2, x3, x4, x
   
    

        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder1 = UpBlock(1024, 512)
        self.decoder2 = UpBlock(512, 256)
        self.decoder3 = UpBlock(256, 128)
        self.decoder4 = UpBlock(128, 64)

    def forward(self, x, x1, x2, x3, x4):
        x = self.decoder1(x, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        return x



class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = ConvolutionBlock(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        dH = skip.size()[2] - x.size()[2]
        dW = skip.size()[3] - x.size()[3]

        x = nn.functional.pad(x, [dW // 2, dW - dW // 2,
                                  dH // 2, dH - dH // 2])

        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvolutionBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        res_conv = self.conv(x)
        res_pool = self.pool(res_conv)

        return res_conv, res_pool




class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

