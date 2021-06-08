import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'tsing_net',
]


class FirstBlock(nn.Module):
    def __init__(self, in_channel, output_channel, stride=1):
        super(FirstBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel,
                               output_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel, eps=9.9999997e-06)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channel,
                               output_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel, eps=9.9999997e-06)

        self.conv3 = nn.Conv2d(in_channel,
                               output_channel,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel, eps=9.9999997e-06)

    def forward(self, x):

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        out2 = self.conv3(x)
        out2 = self.bn3(out2)

        return out1 + out2


class SecondBlock(nn.Module):
    def __init__(self, in_channel, output_channel, stride=1):
        super(SecondBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel,
                               output_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel, eps=9.9999997e-06)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channel,
                               output_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel, eps=9.9999997e-06)

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        return out1 + residual


class TsingNet(nn.Module):
    def __init__(self, num_classes=1000, input_channel=3, loss_type=False):
        super(TsingNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,
                               8,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=9.9999997e-06)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(8, 16)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.loss_type = loss_type
        if self.loss_type:
            self.fc = nn.Linear(128, num_classes, bias=False)
        else:
            self.fc = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, input_channel, output_channel, stride=1):
        return nn.Sequential(
            FirstBlock(input_channel, output_channel, stride=stride),
            nn.ReLU(inplace=True),
            SecondBlock(output_channel, output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        if self.loss_type:
            self.fc.weight = nn.Parameter(
                F.normalize(self.fc.weight, p=2, dim=1))
            return output, self.fc(F.normalize(x, p=2, dim=1))
        else:
            return output


def tsing_net(num_classes=1000, input_channel=3, loss_type=False):

    model = TsingNet(num_classes=num_classes,
                     input_channel=input_channel,
                     loss_type=loss_type)
    return model