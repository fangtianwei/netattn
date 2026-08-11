"""Full NetAttn models for CIFAR-100.

Each residual block exposes shortcut-only and full-block branches. The model
explicitly enumerates all ``2**N`` residual paths, pools their features, and
combines them with learned global attention coefficients before the classifier.

``subnet_idx`` remains checkpoint-compatible. Human-readable path strings use
shallow-to-deep order, with the shallowest residual block at the left.
"""

import torch
import torch.nn as nn


# Stable path-index conversion


def subnet_index_to_bits(index, num_blocks):
    """Convert a stable internal path index to shallow-to-deep path bits.

    Full NetAttn appends all shortcut branches before all full branches at
    every block. Consequently, bit 0 of the internal index represents the
    shallowest residual block. Human-readable path strings instead put the
    shallowest block at the left.
    """
    index = int(index)
    num_blocks = int(num_blocks)
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if index < 0 or index >= 2**num_blocks:
        raise ValueError(f"index must be in [0, {2**num_blocks}), got {index}")
    return format(index, f"0{num_blocks}b")[::-1]


def subnet_bits_to_index(bits):
    """Convert shallow-to-deep path bits to the stable internal path index."""
    cleaned = str(bits).strip().replace(" ", "").replace("_", "")
    if cleaned.startswith(("0b", "0B")):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("bits cannot be empty")
    if any(bit not in ("0", "1") for bit in cleaned):
        raise ValueError(f"bits must contain only 0/1, got {bits!r}")
    return int(cleaned[::-1], 2)


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


# Residual blocks with explicit branch access


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

    def _forward_branches_impl(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        full = out + identity
        full = self.relu(full)
        shortcut = identity

        return shortcut, full

    def forward_branches(self, x):
        """返回当前block的两条子网络分支：shortcut-only和full-block"""
        return self._forward_branches_impl(x)

    def forward(self, x):
        _, full = self._forward_branches_impl(x)
        return full


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

    def _forward_branches_impl(self, x):
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

        full = out + identity
        full = self.relu(full)
        shortcut = identity

        return shortcut, full

    def forward_branches(self, x):
        """返回当前block的两条子网络分支：shortcut-only和full-block"""
        return self._forward_branches_impl(x)

    def forward(self, x):
        _, full = self._forward_branches_impl(x)
        return full


# Full path-enumeration model


class ResNet_NetAttn_CIFAR100(nn.Module):
    """CIFAR-100 ResNet with exact residual-path enumeration.

    The CIFAR stem uses a 3x3 stride-1 convolution without max pooling. For
    ``N = sum(layers)`` residual blocks, each block exposes a shortcut-only
    branch and a full residual branch. The model explicitly enumerates all
    ``2**N`` paths, pools each path feature, and combines them with learned
    global attention weights before classification.

    Compute and memory costs grow exponentially with ``N``, so this
    implementation is intended for shallow networks such as ResNet-18.
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
        super(ResNet_NetAttn_CIFAR100, self).__init__()
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

        self.layers_cfg = layers
        self.num_residual_blocks = sum(layers)
        self.num_subnetworks = 2**self.num_residual_blocks

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

        # 2^N个子网络对应的可学习attention logits
        self.subnetwork_attention_logits = nn.Parameter(
            torch.zeros(self.num_subnetworks)
        )
        self.last_attention_weights = None

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.subnetwork_attention_logits, 0)

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

    def _expand_subnetworks(self, block, subnetworks):
        num_paths, batch_size, channels, height, width = subnetworks.shape
        merged = subnetworks.reshape(num_paths * batch_size, channels, height, width)

        shortcut, full = block.forward_branches(merged)

        output_shape = (num_paths, batch_size) + shortcut.shape[1:]
        shortcut = shortcut.reshape(output_shape)
        full = full.reshape(output_shape)

        # 每个旧路径分裂成两条新路径
        # Preserve the checkpoint-compatible internal order. The current
        # block becomes the next least-significant bit of subnet_idx.
        new_subnetworks = torch.cat([shortcut, full], dim=0)
        return new_subnetworks

    def _enumerate_subnetworks(self, x):
        # 输入: [B, 3, 32, 32]
        x = self.conv1(x)  # [B, 64, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        # 无maxpool

        # 初始只有1条路径（stem输出）
        subnetworks = x.unsqueeze(0)  # [1, B, 64, 32, 32]

        for block in self.layer1:
            subnetworks = self._expand_subnetworks(block, subnetworks)
        for block in self.layer2:
            subnetworks = self._expand_subnetworks(block, subnetworks)
        for block in self.layer3:
            subnetworks = self._expand_subnetworks(block, subnetworks)
        for block in self.layer4:
            subnetworks = self._expand_subnetworks(block, subnetworks)

        return subnetworks

    def _apply_subnetwork_attention(self, subnetworks):
        num_paths, batch_size, channels, height, width = subnetworks.shape
        merged = subnetworks.reshape(num_paths * batch_size, channels, height, width)

        merged = self.avgpool(merged)  # [num_paths * B, C, 1, 1]
        merged = torch.flatten(merged, 1)  # [num_paths * B, C]
        merged = merged.reshape(num_paths, batch_size, -1)  # [num_paths, B, C]

        attention_weights = num_paths * torch.softmax(
            self.subnetwork_attention_logits, dim=0
        )
        attended_feature = torch.sum(
            merged * attention_weights.view(num_paths, 1, 1), dim=0
        )  # [B, C]

        self.last_attention_weights = attention_weights.detach()
        return attended_feature

    def _forward_impl(self, x):
        subnetworks = self._enumerate_subnetworks(x)
        x = self._apply_subnetwork_attention(subnetworks)
        x = self.fc(x)  # [B, num_classes]

        return x

    def get_attention_weights(self):
        """Return cached path weights from the most recent forward pass."""
        return self.last_attention_weights

    def get_subnetwork_strings(self):
        """Return shallow-to-deep bits for every stable ``subnet_idx``.

        A zero selects the shortcut branch and a one selects the full residual
        branch. List position remains compatible with existing checkpoints.
        """
        return [
            subnet_index_to_bits(index, self.num_residual_blocks)
            for index in range(self.num_subnetworks)
        ]

    def forward(self, x):
        return self._forward_impl(x)


# Factory functions


def resnet18_netattn_cifar100(**kwargs):
    """ResNet-18 NetAttn for CIFAR-100"""
    return ResNet_NetAttn_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_netattn_cifar100(**kwargs):
    """ResNet-34 NetAttn for CIFAR-100"""
    return ResNet_NetAttn_CIFAR100(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_netattn_cifar100(**kwargs):
    """ResNet-50 NetAttn for CIFAR-100"""
    return ResNet_NetAttn_CIFAR100(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_netattn_cifar100(**kwargs):
    """ResNet-101 NetAttn for CIFAR-100"""
    return ResNet_NetAttn_CIFAR100(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_netattn_cifar100(**kwargs):
    """ResNet-152 NetAttn for CIFAR-100"""
    return ResNet_NetAttn_CIFAR100(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    # 测试模型
    model = resnet18_netattn_cifar100(num_classes=100)
    print(f"ResNet-18 NetAttn for CIFAR-100")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of residual blocks: {model.num_residual_blocks}")
    print(f"Number of subnetworks: {model.num_subnetworks}")

    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    attention_weights = model.get_attention_weights()
    if attention_weights is not None:
        print(f"Attention shape: {attention_weights.shape}")
        print(f"First 8 attention weights: {attention_weights[:8].tolist()}")
