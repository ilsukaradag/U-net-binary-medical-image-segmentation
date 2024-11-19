import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_blocks_helper import Encoder, Decoder, ConvolutionBlock

class UNet(nn.Module):
    '''
    Initializes the U-Net model, defining the encoder, decoder, and other layers.

    Args:
    - in_channels (int): Number of input channels (1 for scan images).
    - out_channels (int): Number of output channels (1 for binary segmentation masks).
    
    Function:
    - CBR (in_channels, out_channels): Helper function to create a block of Convolution-BatchNorm-ReLU layers. 
    (This function is optional to use)
    '''
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.encoder = Encoder(in_channels)

        self.bottleneck = ConvolutionBlock(512, 1024)

        self.decoder = Decoder()

        self.out =  self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)


      
       
    
    def forward(self, x):
        '''
        Defines the forward pass of the U-Net, performing encoding, bottleneck, and decoding operations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        '''
        x1, x2, x3, x4, x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x, x1, x2, x3, x4)

        x = self.out(x)
        
        return x

