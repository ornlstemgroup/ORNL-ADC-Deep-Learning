
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models import MobileNetV2
class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method
    additionally returns the feature maps at all resolutions for decoder's use.
    """

    def __init__(self, in_channels, norm_layer=None):
        super().__init__()

        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

        # Remove last block
        self.features = self.features[:-1]

        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)

        # Delete classifier
        del self.classifier

    def forward(self, x):
        x0 = x  # 1/1
        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x  # 1/2
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x  # 1/4
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x  # 1/8
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0


class Decoder(nn.Module):
    """
    Decoder upsamples the image by combining the feature maps at all resolutions from the encoder.

    Input:
        x4: (B, C, H/16, W/16) feature map at 1/16 resolution.
        x3: (B, C, H/8, W/8) feature map at 1/8 resolution.
        x2: (B, C, H/4, W/4) feature map at 1/4 resolution.
        x1: (B, C, H/2, W/2) feature map at 1/2 resolution.
        x0: (B, C, H, W) feature map at full resolution.

    Output:
        x: (B, C, H, W) upsampled output at full resolution.
    """

    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        return x

class AtomNet(nn.Module):
    def __init__(self):
        super(AtomNet, self).__init__()

        self.backbone = MobileNetV2Encoder(1)
        self.aspp = ASPP(320, [3, 6, 9])
        self.decoder = Decoder([256, 128, 64, 48, 1], [32, 24, 16, 1])
        self.act = nn.Sigmoid()

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        x4 = self.act(x)
        return x4



if __name__ == "__main__":
    model = AtomNet()
    images1 = torch.zeros([2, 1, 256, 256])
    output = model(images1)
    print(output.shape)