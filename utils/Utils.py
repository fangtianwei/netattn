import torch.nn as nn
import torch
from typing import Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LRScheduler,
)
from collections import Counter

def count_parameters(model: nn.Module):
    """
    计算并打印一个 PyTorch 模型的总参数量和可训练参数量。
    :param model: 需要计算参数的 PyTorch 模型。
    """
    total_params = 0
    trainable_params = 0
        
    for _, parameter in model.named_parameters():
        # parameter.numel() 返回张量中元素的总数
        num_params = parameter.numel()
            
        if parameter.requires_grad:
            trainable_params += num_params
            
        total_params += num_params

    print(f"--- Parameter Count ---")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-------------------------")

# 定义一个函数来计算输出脉冲张量的平均发放率
def cal_firing_rate(s_seq: torch.Tensor):
    # s_seq 的形状是 [T, N, C, H, W] 或 [T, N, L]
    # 我们将其展平为 [T, -1]，然后在时间维度上取平均，得到每个神经元的平均发放率
    # 最后再对所有神经元取平均，得到这一层的平均发放率
    return s_seq.flatten(1).mean(0).mean()


def build_scheduler(optimizer: Optimizer, args: Any) -> (LRScheduler):
    """
    根据 args 中的配置构建并返回学习率调度器。

    支持：
    1) cosa: 纯 CosineAnnealingLR
    2) cosine_warmup: Linear Warmup + CosineAnnealingLR（MaxFormer 风格）
    3) step
    4) multistep
    """
    scheduler_name = getattr(args, 'scheduler', 'cosa')
    print(f"Building scheduler: {scheduler_name}")

    if scheduler_name == 'cosa':
        return CosineAnnealingLR(optimizer, T_max=getattr(args, 'cos_lr_T', 200))

    elif scheduler_name == 'cosine_warmup':
        total_epochs = int(getattr(args, 'epochs', 200))
        warmup_epochs = int(getattr(args, 'warmup_epochs', 20))
        min_lr = float(getattr(args, 'min_lr', 1e-5))

        # 防御性处理，避免里程碑非法
        if warmup_epochs <= 0:
            return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=min_lr)

        if warmup_epochs >= total_epochs:
            warmup_epochs = max(1, total_epochs - 1)

        cosine_t_max = max(1, total_epochs - warmup_epochs)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=min_lr,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    elif scheduler_name == 'step':
        return StepLR(
            optimizer,
            step_size=getattr(args, 'step_size', 64),
            gamma=getattr(args, 'gamma', 0.1),
        )

    elif scheduler_name == 'multistep':
        return MultiStepLR(
            optimizer,
            milestones=getattr(args, 'milestones', [60, 90]),
            gamma=getattr(args, 'gamma', 0.1),
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def calculate_class_weights(trainloader, num_classes):
    """
    计算类别权重，用于解决样本不平衡问题。
    使用逆频率加权方法：样本数量越少的类别权重越高。

    Args:
        trainloader: 训练数据加载器
        num_classes: 类别数量，默认为11（DVS Gesture数据集）

    Returns:
        torch.FloatTensor: 各类别的权重张量，形状为 [num_classes]
    """
    # 统计训练集中的类别分布
    class_counts = Counter()
    for _, targets in trainloader:
        class_counts.update(targets.numpy())

    total_samples = sum(class_counts.values())
    print("类别分布统计:")
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        print(f"  类别 {class_id}: {count} 样本 ({count/total_samples*100:.1f}%)")

    # 计算权重：逆频率加权
    class_weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)  # 避免除零
        weight = total_samples / (num_classes * count)  # 逆频率
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights)
    print(f"类别权重: {class_weights}")
    return class_weights