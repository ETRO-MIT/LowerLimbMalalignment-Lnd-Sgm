# Python packages
import torch
import torchvision
import torch.nn as nn

# Custom functions
from Models.BaseModels import DecoderBlock, ConvRelu


# U-Net architecture
class UNet(nn.Module):
    def __init__(self, number_landmarks: int, pretrained=True):
        super(UNet, self).__init__()
        # Set default parameters
        num_filters = 32
        relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # Create the VGG-16 model from; use ImageNet weights
        if pretrained:
            vgg16_weights = torchvision.models.VGG16_Weights
            encoder = torchvision.models.vgg16(weights=vgg16_weights.IMAGENET1K_V1).features
        else:
            encoder = torchvision.models.vgg16().features

        # Create Encoder path
        self.conv1 = nn.Sequential(encoder[0],
                                   relu,
                                   encoder[2],
                                   relu)

        self.conv2 = nn.Sequential(encoder[5],
                                   relu,
                                   encoder[7],
                                   relu)

        self.conv3 = nn.Sequential(encoder[10],
                                   relu,
                                   encoder[12],
                                   relu,
                                   encoder[14],
                                   relu)

        self.conv4 = nn.Sequential(encoder[17],
                                   relu,
                                   encoder[19],
                                   relu,
                                   encoder[21],
                                   relu)

        self.conv5 = nn.Sequential(encoder[24],
                                   relu,
                                   encoder[26],
                                   relu,
                                   encoder[28],
                                   relu)

        # Create bottleneck
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        # Create decoder
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, number_landmarks, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out = self.final(dec1)

        return x_out
