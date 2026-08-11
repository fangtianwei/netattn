"""Path9 NetAttn configuration for CIFAR-100 ResNet-18.

The nine paths contain every single-skip path plus the all-full path. Execution
is delegated to the shared prefix-DAG implementation.
"""

from typing import Optional, Sequence

import torch

try:
    from .ResNet_NetAttn_Path7_CIFAR100 import (
        BasicBlock,
        Bottleneck,
        ResNet_NetAttn_Path7_CIFAR100,
    )
except ImportError:
    from ResNet_NetAttn_Path7_CIFAR100 import (  # type: ignore
        BasicBlock,
        Bottleneck,
        ResNet_NetAttn_Path7_CIFAR100,
    )


# Default path configuration


DEFAULT_PATH9 = (
    "01111111",
    "10111111",
    "11011111",
    "11101111",
    "11110111",
    "11111011",
    "11111101",
    "11111110",
    "11111111",
)


# Path9 model


class ResNet_NetAttn_Path9_CIFAR100(ResNet_NetAttn_Path7_CIFAR100):
    """
    CIFAR-100 ResNet-18 with NetAttn over skip-one paths plus the all-full path.

    This reuses the Path7 prefix-DAG implementation and changes only the
    default path set. Bit strings are interpreted left-to-right as L1.B0,
    L1.B1, L2.B0, L2.B1, L3.B0, L3.B1, L4.B0, L4.B1.
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
        if path_strings is None:
            path_strings = DEFAULT_PATH9
        super().__init__(
            block=block,
            layers=layers,
            in_channels=in_channels,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            path_strings=path_strings,
            attention_total_budget=attention_total_budget,
        )


# Factory and compatibility alias


def resnet18_netattn_path9_cifar100(**kwargs):
    return ResNet_NetAttn_Path9_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


ResNet_NetAttn_PathSet_CIFAR100 = ResNet_NetAttn_Path9_CIFAR100


if __name__ == "__main__":
    model = resnet18_netattn_path9_cifar100(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    print("Paths:", model.get_subnetwork_strings())
    print("Attention:", model.get_attention_weights())
    print("Prefix schedule:", model.get_prefix_schedule())
