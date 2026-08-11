"""Pattern-configurable attention for the Full NetAttn model.

Patterns use shallow-to-deep ``0``/``1``/``X`` strings. They can initialize
trainable attention coefficients or hold selected path coefficients fixed while
the remaining paths share the available attention budget through softmax.
"""

import torch
from collections.abc import Sequence
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union, cast

try:
    from .ResNet_NetAttn_CIFAR100 import (
        BasicBlock,
        Bottleneck,
        ResNet_NetAttn_CIFAR100,
        subnet_bits_to_index,
        subnet_index_to_bits,
    )
except ImportError:
    from ResNet_NetAttn_CIFAR100 import (  # type: ignore
        BasicBlock,
        Bottleneck,
        ResNet_NetAttn_CIFAR100,
        subnet_bits_to_index,
        subnet_index_to_bits,
    )


# Public pattern specification types


PatternGroup = Union[str, Sequence[str]]
PatternAttentionSpec = Optional[
    Union[
        Dict[str, float],
        List[Tuple[PatternGroup, float]],
        Tuple[Tuple[PatternGroup, float], ...],
    ]
]


# Configurable Full NetAttn


class ResNet_NetAttn_Config_CIFAR100(ResNet_NetAttn_CIFAR100):
    """
    在原始 NetAttn 的基础上，为 2^N 条子网路径提供两类可控接口：

    1. 初始化权重：
       只在训练开始前改一次 attention 初始值，后续仍继续学习。

    2. 固定权重：
       将某些子网的 attention 权重固定为指定值，训练中始终不变；
       剩余子网继续按 softmax 分配剩余预算。

    统一使用 0 / 1 / X 三种字符描述 bit 模式：
    - 0: 该位必须为 0
    - 1: 该位必须为 1
    - X: 该位任意

    例如在 ResNet-18 的 8-bit 子网中：
    - "00XXXXXX" : 前两位为 0
    - "XXXXXX11" : 后两位为 1
    - "1X0X1X0X" : 中间任意混合约束

    也支持多个 pattern 一起输入，例如：
    [
        (["00XXXXXX", "XXXXXX11"], 0.0),
        ("11111111", 12.0),
    ]

    规则说明：
    - 若多个规则有重叠，则后面的规则覆盖前面的规则。
    - 固定权重模式下，固定部分不参与反向传播；剩余部分照常训练。
    - 初始化模式下，本质上仍是通过 logits 初始化实现，因此 softmax 路径必须保持正值。
      如果你把某些路径初始化为 0，会自动替换成一个很小的正数 `min_init_attention`，
      以保证这些路径后续仍然可以被学习。若想“严格为 0 且永远不变”，应使用固定权重模式。
    - `attention_total_budget` 用来控制 attention 总和；默认等于 `num_subnetworks`，
      但也可以手动设成 1.0、64.0 等任意正数，以支持更一般的缩放实验。
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
        init_pattern_attention_spec: PatternAttentionSpec = None,
        fixed_pattern_attention_spec: PatternAttentionSpec = None,
        min_init_attention: float = 1e-3,
        attention_total_budget: Optional[float] = None,
        enforce_attention_total_budget: bool = True,
    ):
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
        )

        self.subnetwork_bit_strings = self.get_subnetwork_strings()
        self.min_init_attention = float(min_init_attention)
        self.default_attention_total_budget = float(self.num_subnetworks)
        self.enforce_attention_total_budget = bool(enforce_attention_total_budget)
        if attention_total_budget is None:
            self.attention_total_budget = self.default_attention_total_budget
        else:
            self.attention_total_budget = float(attention_total_budget)
        if self.attention_total_budget <= 0:
            raise ValueError("attention_total_budget must be positive")

        self.register_buffer(
            "fixed_attention_mask",
            torch.zeros(self.num_subnetworks, dtype=torch.bool),
        )
        self.register_buffer(
            "fixed_attention_values",
            torch.zeros(
                self.num_subnetworks,
                dtype=self.subnetwork_attention_logits.dtype,
            ),
        )

        self.configure_fixed_attention(
            fixed_pattern_attention_spec=fixed_pattern_attention_spec,
        )

        if init_pattern_attention_spec is not None:
            self.configure_initial_attention(
                init_pattern_attention_spec=init_pattern_attention_spec,
                min_init_attention=self.min_init_attention,
            )

    def _fixed_attention_mask_buffer(self) -> Tensor:
        return cast(Tensor, getattr(self, "fixed_attention_mask"))

    def _fixed_attention_values_buffer(self) -> Tensor:
        return cast(Tensor, getattr(self, "fixed_attention_values"))

    def get_fixed_attention_mask(self) -> Tensor:
        return self._fixed_attention_mask_buffer().detach().clone()

    def get_fixed_attention_values(self) -> Tensor:
        return self._fixed_attention_values_buffer().detach().clone()

    def get_attention_total_budget(self) -> float:
        return float(self.attention_total_budget)

    def get_enforce_attention_total_budget(self) -> bool:
        return bool(self.enforce_attention_total_budget)

    def _spec_to_pairs(self, spec, spec_name: str):
        if spec is None:
            return []
        if isinstance(spec, dict):
            return list(spec.items())
        if isinstance(spec, (list, tuple)):
            pairs = []
            for item in spec:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise TypeError(
                        f"{spec_name} must be a dict or a list/tuple of (pattern, value) pairs"
                    )
                pairs.append((item[0], item[1]))
            return pairs
        raise TypeError(
            f"{spec_name} must be a dict or a list/tuple of (pattern, value) pairs"
        )

    def _normalize_pattern_group(self, pattern_group: PatternGroup) -> List[str]:
        if isinstance(pattern_group, str):
            return [pattern_group]

        if isinstance(pattern_group, Sequence):
            patterns = []
            for pattern in pattern_group:
                if not isinstance(pattern, str):
                    raise TypeError(
                        "each pattern inside a pattern group must be a string"
                    )
                patterns.append(pattern)
            if len(patterns) == 0:
                raise ValueError("pattern group cannot be empty")
            return patterns

        raise TypeError("pattern must be either a string or a sequence of strings")

    def _clean_pattern_string(self, pattern: str) -> str:
        cleaned = pattern.strip().replace(" ", "").replace("_", "").upper()
        if cleaned.startswith("0B"):
            cleaned = cleaned[2:]
        return cleaned

    def _validate_pattern_string(self, pattern: str) -> str:
        cleaned = self._clean_pattern_string(pattern)
        if len(cleaned) != self.num_residual_blocks:
            raise ValueError(
                f"pattern length must be {self.num_residual_blocks}, got {cleaned}"
            )
        if any(bit not in ("0", "1", "X") for bit in cleaned):
            raise ValueError(
                f"pattern must contain only '0', '1', or 'X', got {cleaned}"
            )
        return cleaned

    def _pattern_matches_bits(self, pattern: str, bits: str) -> bool:
        return all(
            pattern_bit == "X" or pattern_bit == bit
            for pattern_bit, bit in zip(pattern, bits)
        )

    def _pattern_to_mask(self, pattern: str) -> Tensor:
        validated_pattern = self._validate_pattern_string(pattern)
        matched = [
            self._pattern_matches_bits(validated_pattern, bits)
            for bits in self.subnetwork_bit_strings
        ]
        return torch.tensor(
            matched,
            dtype=torch.bool,
            device=self.subnetwork_attention_logits.device,
        )

    def _build_pattern_value_tensors(
        self,
        pattern_attention_spec: PatternAttentionSpec,
        spec_name: str,
    ) -> Tuple[Tensor, Tensor]:
        device = self.subnetwork_attention_logits.device
        dtype = self.subnetwork_attention_logits.dtype

        selected_mask = torch.zeros(
            self.num_subnetworks, dtype=torch.bool, device=device
        )
        selected_values = torch.zeros(self.num_subnetworks, dtype=dtype, device=device)

        for pattern_group, value in self._spec_to_pairs(
            pattern_attention_spec, spec_name
        ):
            pattern_value = float(value)
            if pattern_value < 0:
                raise ValueError(
                    f"{spec_name} contains a negative value: {pattern_value}"
                )

            for pattern in self._normalize_pattern_group(pattern_group):
                pattern_mask = self._pattern_to_mask(pattern)
                selected_mask = torch.logical_or(selected_mask, pattern_mask)
                selected_values = torch.where(
                    pattern_mask,
                    torch.full_like(selected_values, pattern_value),
                    selected_values,
                )

        return selected_mask, selected_values

    def reset_attention_logits_to_uniform(self):
        """将 attention logits 重置为全 0，对应均匀 attention。"""
        with torch.no_grad():
            self.subnetwork_attention_logits.zero_()

    def configure_initial_attention(
        self,
        init_pattern_attention_spec: PatternAttentionSpec,
        min_init_attention: Optional[float] = None,
    ):
        """
        按 0/1/X pattern 设置“初始化 attention 权重”。

        这个接口只影响初始 logits，后续训练仍会更新这些路径。

        示例：
        - {"00XXXXXX": 0.0}
          将前两位为 0 的路径初始化到接近 0 的注意力，然后继续学习。
        - [(["00XXXXXX", "XXXXXX11"], 0.0), ("11111111", 8.0)]
          多个模式一起设置。

        说明：
        - 若未被指定的路径仍存在，则剩余预算会在这些路径上均匀分配。
        - 若某些路径被指定为 0，因为 softmax 不能直接表示严格 0，
          会自动替换为一个很小的正数 `min_init_attention`。
        - 若想严格固定为 0 且不再学习，请使用 `configure_fixed_attention()`。
        """
        if min_init_attention is None:
            min_init_attention = self.min_init_attention
        min_init_attention = float(min_init_attention)
        if min_init_attention <= 0:
            raise ValueError("min_init_attention must be positive")

        init_mask, init_values = self._build_pattern_value_tensors(
            init_pattern_attention_spec,
            "init_pattern_attention_spec",
        )

        target_attention = init_values.clone()
        specified_total = float(init_values.sum().item())
        if specified_total > self.attention_total_budget + 1e-6:
            raise ValueError(
                f"sum of initialized attention values ({specified_total}) must not exceed attention_total_budget ({self.attention_total_budget})"
            )

        unspecified_mask = torch.logical_not(init_mask)
        unspecified_count = int(unspecified_mask.sum().item())
        remaining_budget = self.attention_total_budget - specified_total

        if unspecified_count > 0:
            target_attention[unspecified_mask] = remaining_budget / unspecified_count
        elif abs(remaining_budget) > 1e-6:
            raise ValueError(
                "all initial attention entries are specified, so their sum must exactly equal attention_total_budget"
            )

        zero_mask = target_attention <= 0
        if bool(zero_mask.any().item()):
            zero_count = int(zero_mask.sum().item())
            positive_mask = torch.logical_not(zero_mask)
            positive_total = float(target_attention[positive_mask].sum().item())
            floor_total = min_init_attention * zero_count

            if positive_total <= 0:
                raise ValueError(
                    "initial attention cannot make all paths zero; keep some positive budget or use fixed attention mode"
                )
            if floor_total >= self.attention_total_budget:
                raise ValueError(
                    "min_init_attention is too large for the current initialization pattern"
                )

            scale = (self.attention_total_budget - floor_total) / positive_total
            target_attention[positive_mask] = target_attention[positive_mask] * scale
            target_attention[zero_mask] = min_init_attention

        target_sum = float(target_attention.sum().item())
        if target_sum <= 0:
            raise ValueError("target initial attention sum must be positive")
        target_attention = target_attention * (self.attention_total_budget / target_sum)

        target_logits = torch.log(target_attention)
        target_logits = target_logits - target_logits.mean()

        with torch.no_grad():
            self.subnetwork_attention_logits.copy_(target_logits)

    def configure_fixed_attention(
        self,
        fixed_pattern_attention_spec: PatternAttentionSpec,
    ):
        """
        按 0/1/X pattern 设置“固定 attention 权重”。

        固定后的路径在训练期间不会改变；其余路径继续按 softmax 分配剩余预算。

        示例：
        - {"00XXXXXX": 0.0}
          前两位为 0 的路径全部固定为 0。
        - {"XXXXXX11": 2.0}
          所有后两位为 1 的路径，每条固定为 2.0。
        - [(["00XXXXXX", "XXXXXX11"], 0.0), ("11111111", 10.0)]
          支持多个 pattern 同时输入。

        说明：
        - 若多条规则重叠，后面的规则覆盖前面的规则。
        - 默认 enforce_attention_total_budget=True：固定值之和不能超过 attention_total_budget；
          若全部路径都被固定，则固定值总和必须恰好等于 attention_total_budget。
        - 若 enforce_attention_total_budget=False：固定值按原值直接使用，不再强制匹配 total budget；
          适合测试时只保留 TopK 训练得到的原始 attention value。
        """
        fixed_mask, fixed_values = self._build_pattern_value_tensors(
            fixed_pattern_attention_spec,
            "fixed_pattern_attention_spec",
        )

        fixed_total = float(fixed_values.sum().item())
        if self.enforce_attention_total_budget:
            if fixed_total > self.attention_total_budget + 1e-6:
                raise ValueError(
                    f"sum of fixed attention values ({fixed_total}) must not exceed attention_total_budget ({self.attention_total_budget})"
                )

            trainable_mask = torch.logical_not(fixed_mask)
            has_trainable_paths = bool(trainable_mask.any().item())
            if (not has_trainable_paths) and abs(
                fixed_total - self.attention_total_budget
            ) > 1e-6:
                raise ValueError(
                    "all attention entries are fixed, so their sum must exactly equal attention_total_budget"
                )

        fixed_attention_mask = self._fixed_attention_mask_buffer()
        fixed_attention_values = self._fixed_attention_values_buffer()
        fixed_attention_mask.copy_(fixed_mask)
        fixed_attention_values.copy_(fixed_values)

    def clear_fixed_attention(self):
        """清除全部固定 attention 规则，恢复原始可学习 attention 分配。"""
        self.configure_fixed_attention(fixed_pattern_attention_spec=None)

    def get_pattern_matched_indices(self, pattern: str) -> List[int]:
        """返回某个 0/1/X 模式命中的子网索引列表。"""
        mask = self._pattern_to_mask(pattern)
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()

    def _apply_subnetwork_attention(self, subnetworks):
        num_paths, batch_size, channels, height, width = subnetworks.shape
        merged = subnetworks.reshape(num_paths * batch_size, channels, height, width)

        merged = self.avgpool(merged)  # [num_paths * B, C, 1, 1]
        merged = torch.flatten(merged, 1)  # [num_paths * B, C]
        merged = merged.reshape(num_paths, batch_size, -1)  # [num_paths, B, C]

        if num_paths != self.num_subnetworks:
            raise ValueError(
                f"num_paths ({num_paths}) does not match initialized num_subnetworks ({self.num_subnetworks})"
            )

        fixed_mask = self._fixed_attention_mask_buffer()
        fixed_values = self._fixed_attention_values_buffer()
        trainable_mask = torch.logical_not(fixed_mask)

        fixed_total = float(fixed_values.sum().item())
        attention_weights = fixed_values.clone()

        if bool(trainable_mask.any().item()):
            if self.enforce_attention_total_budget:
                trainable_budget_value = self.attention_total_budget - fixed_total
                if trainable_budget_value < -1e-6:
                    raise ValueError(
                        f"remaining attention budget is negative: {trainable_budget_value:.6f}"
                    )
            else:
                # 不限制 total budget 时，fixed values 原样使用；仍给未固定路径一个独立 softmax budget。
                trainable_budget_value = self.attention_total_budget

            trainable_logits = self.subnetwork_attention_logits[trainable_mask]
            trainable_budget = torch.tensor(
                trainable_budget_value,
                dtype=self.subnetwork_attention_logits.dtype,
                device=self.subnetwork_attention_logits.device,
            )
            trainable_weights = trainable_budget * torch.softmax(
                trainable_logits, dim=0
            )
            attention_weights[trainable_mask] = trainable_weights
        elif self.enforce_attention_total_budget:
            remaining_budget_value = self.attention_total_budget - fixed_total
            if abs(remaining_budget_value) > 1e-6:
                raise ValueError(
                    "no trainable paths remain, but remaining attention budget is not zero"
                )

        attended_feature = torch.sum(
            merged * attention_weights.view(num_paths, 1, 1), dim=0
        )  # [B, C]

        self.last_attention_weights = attention_weights.detach()
        return attended_feature


# Factory functions


def resnet18_netattn_config_cifar100(**kwargs):
    """ResNet-18 NetAttn，支持 pattern-based 初始化和固定 attention。"""
    return ResNet_NetAttn_Config_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_netattn_config_firsttwozero_cifar100(**kwargs):
    """ResNet-18 NetAttn，默认将前两位 bit 为 0 的路径固定为 0。"""
    kwargs.setdefault("fixed_pattern_attention_spec", {"00XXXXXX": 0.0})
    return ResNet_NetAttn_Config_CIFAR100(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_netattn_config_cifar100(**kwargs):
    """ResNet-34 NetAttn，支持 pattern-based 初始化和固定 attention。"""
    return ResNet_NetAttn_Config_CIFAR100(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_netattn_config_cifar100(**kwargs):
    """ResNet-50 NetAttn，支持 pattern-based 初始化和固定 attention。"""
    return ResNet_NetAttn_Config_CIFAR100(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_netattn_config_cifar100(**kwargs):
    """ResNet-101 NetAttn，支持 pattern-based 初始化和固定 attention。"""
    return ResNet_NetAttn_Config_CIFAR100(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_netattn_config_cifar100(**kwargs):
    """ResNet-152 NetAttn，支持 pattern-based 初始化和固定 attention。"""
    return ResNet_NetAttn_Config_CIFAR100(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == "__main__":
    model = resnet18_netattn_config_cifar100(
        num_classes=100,
        init_pattern_attention_spec={"00XXXXXX": 0.0},
        fixed_pattern_attention_spec={"XXXXXX11": 0.0},
    )

    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    attention_weights = model.get_attention_weights()
    if attention_weights is None:
        raise RuntimeError("attention weights are not available after forward pass")

    fixed_mask = model.get_fixed_attention_mask()
    fixed_values = model.get_fixed_attention_values()

    print("ResNet-18 Config NetAttn for CIFAR-100")
    print(f"Output shape: {y.shape}")
    print(f"Attention sum: {attention_weights.sum().item():.4f}")
    print(f"Fixed path count: {fixed_mask.sum().item()}")
    print(f"Fixed attention sum: {fixed_values.sum().item():.4f}")
    print(
        f"Matched by 00XXXXXX: {model.get_pattern_matched_indices('00XXXXXX')[:8]} ..."
    )
    print(f"First 8 attention weights: {attention_weights[:8].tolist()}")
