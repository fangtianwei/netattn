"""CIFAR-100 ResNet baselines.

The stem is adapted to 32x32 inputs: a 3x3 stride-1 convolution is used
without an ImageNet-style max-pooling layer. Spatial downsampling happens in
the first block of layers 2-4 through stride-2 convolutions.
"""

import torch
import torch.nn as nn


# Convolution helpers


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Residual blocks


class BasicBlock(nn.Module):
    """标准BasicBlock，使用conv stride=2进行下采样"""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # conv1: 使用stride进行下采样（stride=2时）
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for deeper networks (ResNet-50/101/152)"""

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# CIFAR-100 ResNet


class ResNet_CIFAR100(nn.Module):
    """
    ResNet for CIFAR-100 (Baseline)

    与ImageNet版本的主要区别：
    1. 首层卷积: kernel_size=3, stride=1, padding=1 (保持32x32分辨率)
    2. 无首层MaxPool (CIFAR图像太小)
    3. 最终特征图: 4x4 (32 -> 16 -> 8 -> 4)
    """

    def __init__(
        self,
        block,
        layers,
        in_channels=3,
        num_classes=100,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet_CIFAR100, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR版本: kernel_size=3, stride=1, padding=1
        # 保持输入32x32的分辨率
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 注意: CIFAR版本不使用maxpool，因为输入图像已经很小

        # 构建4个layer
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize残差分支的最后一个BN
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 输入: [B, 3, 32, 32]
        x = self.conv1(x)  # [B, 64, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        # 无maxpool

        x = self.layer1(x)  # [B, 64, 32, 32]
        x = self.layer2(x)  # [B, 128, 16, 16]
        x = self.layer3(x)  # [B, 256, 8, 8]
        x = self.layer4(x)  # [B, 512, 4, 4]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)  # [B, num_classes]

        return x

    def forward(self, x):
        return self._forward_impl(x)


# Factory functions


def resnet18_cifar100(**kwargs):
    """ResNet-18 for CIFAR-100"""
    return ResNet_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_cifar100(**kwargs):
    """ResNet-34 for CIFAR-100"""
    return ResNet_CIFAR100(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_cifar100(**kwargs):
    """ResNet-50 for CIFAR-100"""
    return ResNet_CIFAR100(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_cifar100(**kwargs):
    """ResNet-101 for CIFAR-100"""
    return ResNet_CIFAR100(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_cifar100(**kwargs):
    """ResNet-152 for CIFAR-100"""
    return ResNet_CIFAR100(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    # 测试模型
    model = resnet18_cifar100(num_classes=100)
    print(f"ResNet-18 for CIFAR-100")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
