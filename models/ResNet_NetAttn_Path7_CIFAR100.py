"""Fixed-path NetAttn with prefix-DAG execution for CIFAR-100.

Only prefixes that can reach a configured final path are retained after each
residual block. Path strings are shallow-to-deep: the leftmost bit controls
``L1.B0`` and the rightmost bit controls ``L4.B1``.
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# Default path configuration


DEFAULT_PATH7 = (
    "00111111",
    "11111111",
    "11111110",
    "11111101",
    "11110111",
    "11110101",
    "11110110",
)


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
    """Standard BasicBlock using conv stride for downsampling."""

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
        """Return shortcut-only and full-block branches for this block."""
        return self._forward_branches_impl(x)

    def forward(self, x):
        _, full = self._forward_branches_impl(x)
        return full


class Bottleneck(nn.Module):
    """Bottleneck block kept for API symmetry; Path7 factory uses BasicBlock."""

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
        """Return shortcut-only and full-block branches for this block."""
        return self._forward_branches_impl(x)

    def forward(self, x):
        _, full = self._forward_branches_impl(x)
        return full


# Prefix-DAG path model


class ResNet_NetAttn_Path7_CIFAR100(nn.Module):
    """
    CIFAR-100 ResNet-18 with NetAttn over a fixed path set.

    The path set is executed as a prefix DAG: after each residual block, only
    prefixes that can still reach one of ``path_strings`` are retained. Bit
    strings are interpreted left-to-right as L1.B0, L1.B1, L2.B0, L2.B1,
    L3.B0, L3.B1, L4.B0, L4.B1.
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
        path_strings: Optional[Sequence[str]] = None,
        attention_total_budget=1.0,
    ):
        super(ResNet_NetAttn_Path7_CIFAR100, self).__init__()
        if list(layers) != [2, 2, 2, 2]:
            raise ValueError("Path7 model supports only ResNet-18 layers [2, 2, 2, 2]")
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

        self.layers_cfg = list(layers)
        self.num_residual_blocks = sum(layers)
        self.path_strings = self._normalize_path_strings(path_strings)
        self.num_paths = len(self.path_strings)
        self.num_subnetworks = self.num_paths
        self.attention_total_budget = float(attention_total_budget)
        if self.attention_total_budget <= 0:
            raise ValueError("attention_total_budget must be positive")

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

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

        self.path_attention_logits = nn.Parameter(torch.zeros(self.num_paths))
        self.last_attention_weights = None
        self.block_names = self._compute_block_names()
        self.prefix_schedule = self._build_prefix_schedule()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.path_attention_logits, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _normalize_path_strings(self, path_strings):
        if path_strings is None:
            path_strings = DEFAULT_PATH7
        normalized = []
        seen = set()
        for bits in path_strings:
            cleaned = str(bits).strip().replace(" ", "").replace("_", "")
            if cleaned.startswith("0b") or cleaned.startswith("0B"):
                cleaned = cleaned[2:]
            if len(cleaned) != self.num_residual_blocks:
                raise ValueError(
                    "each path bit string must have length "
                    f"{self.num_residual_blocks}, got {bits}"
                )
            if any(bit not in ("0", "1") for bit in cleaned):
                raise ValueError(f"path bit string must contain only 0/1, got {bits}")
            if cleaned in seen:
                raise ValueError(f"duplicate path bit string: {cleaned}")
            seen.add(cleaned)
            normalized.append(cleaned)
        if not normalized:
            raise ValueError("path_strings cannot be empty")
        return tuple(normalized)

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

    def _iter_residual_blocks(self):
        for layer_index, layer in enumerate(
            (self.layer1, self.layer2, self.layer3, self.layer4), start=1
        ):
            for block_index, block in enumerate(layer):
                yield f"L{layer_index}.B{block_index}", block

    def _compute_block_names(self):
        return [name for name, _ in self._iter_residual_blocks()]

    def _build_prefix_schedule(self):
        schedule = []
        for prefix_len in range(1, self.num_residual_blocks + 1):
            prefixes = []
            seen = set()
            for bits in self.path_strings:
                prefix = bits[:prefix_len]
                if prefix not in seen:
                    seen.add(prefix)
                    prefixes.append(prefix)
            schedule.append(
                {
                    "block": self.block_names[prefix_len - 1],
                    "prefix_len": prefix_len,
                    "prefixes": prefixes,
                }
            )
        return schedule

    def _next_prefixes_by_parent(self, block_position: int) -> Dict[str, List[str]]:
        next_prefixes = self.prefix_schedule[block_position]["prefixes"]
        parent_to_children: Dict[str, List[str]] = {}
        for prefix in next_prefixes:
            parent = prefix[:-1]
            parent_to_children.setdefault(parent, []).append(prefix)
        return parent_to_children

    def _expand_prefix_features(self, block, prefix_features, block_position):
        parent_to_children = self._next_prefixes_by_parent(block_position)
        ordered_parents = [
            prefix for prefix in prefix_features if prefix in parent_to_children
        ]
        if not ordered_parents:
            raise RuntimeError(f"no active prefixes before block {block_position}")

        stacked = torch.stack([prefix_features[prefix] for prefix in ordered_parents])
        num_prefixes, batch_size, channels, height, width = stacked.shape
        merged = stacked.reshape(num_prefixes * batch_size, channels, height, width)

        shortcut, full = block.forward_branches(merged)

        output_shape = (num_prefixes, batch_size) + shortcut.shape[1:]
        shortcut = shortcut.reshape(output_shape)
        full = full.reshape(output_shape)

        next_features = {}
        for parent_position, parent in enumerate(ordered_parents):
            for child in parent_to_children[parent]:
                branch = full if child[-1] == "1" else shortcut
                next_features[child] = branch[parent_position]
        return next_features

    def _enumerate_path_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        prefix_features = {"": x}
        for block_position, (_, block) in enumerate(self._iter_residual_blocks()):
            prefix_features = self._expand_prefix_features(
                block=block,
                prefix_features=prefix_features,
                block_position=block_position,
            )

        missing = [bits for bits in self.path_strings if bits not in prefix_features]
        if missing:
            raise RuntimeError(f"missing final path features: {missing}")
        return torch.stack([prefix_features[bits] for bits in self.path_strings])

    def _apply_path_attention(self, path_features):
        num_paths, batch_size, channels, height, width = path_features.shape
        merged = path_features.reshape(num_paths * batch_size, channels, height, width)

        merged = self.avgpool(merged)
        merged = torch.flatten(merged, 1)
        merged = merged.reshape(num_paths, batch_size, -1)

        attention_weights = self.attention_total_budget * torch.softmax(
            self.path_attention_logits, dim=0
        )
        attended_feature = torch.sum(
            merged * attention_weights.view(num_paths, 1, 1), dim=0
        )

        self.last_attention_weights = attention_weights.detach()
        return attended_feature

    def _forward_impl(self, x):
        path_features = self._enumerate_path_features(x)
        x = self._apply_path_attention(path_features)
        x = self.fc(x)
        return x

    def get_attention_weights(self):
        """Return cached path weights from the most recent forward pass."""
        return self.last_attention_weights

    def get_attention_total_budget(self) -> float:
        return float(self.attention_total_budget)

    def get_path_strings(self):
        return list(self.path_strings)

    def get_subnetwork_strings(self):
        return self.get_path_strings()

    def get_prefix_schedule(self):
        return [
            {
                "block": row["block"],
                "prefix_len": row["prefix_len"],
                "prefixes": list(row["prefixes"]),
            }
            for row in self.prefix_schedule
        ]

    def _skip_blocks_for_bits(self, bits: str):
        return [
            self.block_names[position]
            for position, bit in enumerate(bits)
            if bit == "0"
        ]

    def get_path_table_rows(self):
        """Return configured paths in their stable model order."""
        weights = self.last_attention_weights
        if weights is None:
            weights = self.attention_total_budget * torch.softmax(
                self.path_attention_logits.detach(), dim=0
            )
        weights = weights.detach().cpu()
        logits = self.path_attention_logits.detach().cpu()

        rows = []
        for rank, bits in enumerate(self.path_strings, start=1):
            rows.append(
                {
                    "rank": rank,
                    "bits": bits,
                    "skip_blocks": self._skip_blocks_for_bits(bits),
                    "logit": float(logits[rank - 1].item()),
                    "attention_weight": float(weights[rank - 1].item()),
                }
            )
        return rows

    def forward(self, x):
        return self._forward_impl(x)


# Factory and compatibility alias


def resnet18_netattn_path7_cifar100(**kwargs):
    return ResNet_NetAttn_Path7_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


ResNet_NetAttn_PathSet_CIFAR100 = ResNet_NetAttn_Path7_CIFAR100


if __name__ == "__main__":
    model = resnet18_netattn_path7_cifar100(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    print("Paths:", model.get_subnetwork_strings())
    print("Attention:", model.get_attention_weights())
    print("Prefix schedule:", model.get_prefix_schedule())
