import torch.nn as nn
from utilities.conv2d_mtl import Conv2dMtl


class DnCNNMtl(nn.Module):
    def __init__(self, channels, num_of_layers=17,mtl=True):
        super(DnCNNMtl, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        if mtl:
            self.conv2d = Conv2dMtl
        else:
            self.conv2d = nn.Conv2d
        layers = []
        layers.append(self.conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(self.conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(self.conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, self.conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.dncnn(x)
        return out
