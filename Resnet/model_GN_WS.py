import torch.nn as nn
import torch
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    """Weight Standardization 卷积层"""

    def forward(self, x):
        weight = self.weight
        # 计算权重均值 (按输出通道维度)
        weight_mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        # 中心化
        weight = weight - weight_mean
        # 计算标准差
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        # 标准化
        weight = weight / std.expand_as(weight)

        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 使用 WSConv2d 替换标准卷积
        self.conv1 = WSConv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用 GroupNorm 替换 BatchNorm,分组数=通道数的因子
        num_groups = self._get_num_groups(out_channel)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channel)
        self.relu = nn.ReLU()

        self.conv2 = WSConv2d(in_channels=out_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=1, bias=False)
        num_groups = self._get_num_groups(out_channel)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channel)
        self.downsample = downsample

    def _get_num_groups(self, channels):
        """根据通道数选择合适的分组数"""
        # 优先选择32组,如果通道数不够则选择较小的因子
        if channels % 32 == 0:
            return 32
        elif channels % 16 == 0:
            return 16
        elif channels % 8 == 0:
            return 8
        else:
            return channels  # 最坏情况:每个通道一组(等同于LayerNorm)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_GN_WS(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck_GN_WS, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = WSConv2d(in_channels=in_channel, out_channels=width,
                              kernel_size=1, stride=1, bias=False)
        num_groups = self._get_num_groups(width)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=width)

        self.conv2 = WSConv2d(in_channels=width, out_channels=width, groups=groups,
                              kernel_size=3, stride=stride, bias=False, padding=1)
        num_groups = self._get_num_groups(width)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=width)

        self.conv3 = WSConv2d(in_channels=width, out_channels=out_channel * self.expansion,
                              kernel_size=1, stride=1, bias=False)
        num_groups = self._get_num_groups(out_channel * self.expansion)
        self.gn3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _get_num_groups(self, channels):
        if channels % 32 == 0:
            return 32
        elif channels % 16 == 0:
            return 16
        elif channels % 8 == 0:
            return 8
        else:
            return channels

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=10, include_top=True,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group

        # CIFAR-10专用stem:3x3卷积,stride=1,无maxpool
        self.conv1 = WSConv2d(3, self.in_channel, kernel_size=3, stride=1,
                              padding=1, bias=False)
        num_groups = 32 if self.in_channel % 32 == 0 else 16
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            # 权重初始化
        for m in self.modules():
            if isinstance(m, WSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            num_groups = 32 if (channel * block.expansion) % 32 == 0 else 16
            downsample = nn.Sequential(
                WSConv2d(self.in_channel, channel * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups,
                             num_channels=channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        # 注意:CIFAR-10版本没有maxpool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x



def resnet18_gn_ws(num_classes=10, include_top=True):
    """ResNet-18 with GN+WS (11M参数)"""
    return ResNet(BasicBlock, [2, 2, 2, 2],
                       num_classes=num_classes, include_top=include_top)

# 或者在现有模型基础上减少层数
def resnet34_gn_ws(num_classes=10, include_top=True):
    """ResNet-34 with GN+WS (21M参数)"""
    return ResNet(BasicBlock, [3, 4, 6, 3],
                        num_classes=num_classes, include_top=include_top)


def resnet50_gn_ws(num_classes=10, include_top=True):
    """ResNet-50 with GN+WS (23M参数)"""
    return ResNet(Bottleneck_GN_WS, [3, 4, 6, 3],
                        num_classes=num_classes, include_top=include_top)