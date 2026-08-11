# datasets/cifar100_da_dataset.py
# CIFAR-100 数据集 - 带高级数据增强版本 (Data Augmentation)
# 包含: RandAugment, Mixup, CutMix, Random Erasing, Label Smoothing

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# 尝试导入timm的数据增强工具
try:
    from timm.data.auto_augment import rand_augment_transform
    from timm.data import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm not installed. Advanced augmentation will be limited.")
    print("Install with: pip install timm")


class MixupCutmix:
    """
    Mixup和CutMix数据增强类
    
    在训练循环中使用，对batch进行混合增强
    """
    def __init__(self, mixup_alpha=0.75, cutmix_alpha=0.5, prob=1.0, 
                 switch_prob=0.5, label_smoothing=0.1, num_classes=100):
        """
        Args:
            mixup_alpha: Mixup的Beta分布参数
            cutmix_alpha: CutMix的Beta分布参数
            prob: 应用增强的概率
            switch_prob: 切换到CutMix的概率
            label_smoothing: 标签平滑系数
            num_classes: 类别数量
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
    def __call__(self, x, y):
        """
        Args:
            x: 输入图像 (batch, C, H, W)
            y: 标签 (batch,)
        Returns:
            mixed_x: 混合后的图像
            y_mixed: 混合后的软标签
        """
        if np.random.rand() > self.prob:
            # 不进行混合，只做标签平滑
            return x, self._smooth_labels(y)
        
        # 决定使用Mixup还是CutMix
        use_cutmix = np.random.rand() < self.switch_prob
        
        if use_cutmix and self.cutmix_alpha > 0:
            return self._cutmix(x, y)
        elif self.mixup_alpha > 0:
            return self._mixup(x, y)
        else:
            return x, self._smooth_labels(y)
    
    def _smooth_labels(self, y):
        """将硬标签转换为软标签"""
        batch_size = y.size(0)
        soft_labels = torch.zeros(batch_size, self.num_classes, device=y.device)
        soft_labels.fill_(self.label_smoothing / (self.num_classes - 1))
        soft_labels.scatter_(1, y.unsqueeze(1), 1.0 - self.label_smoothing)
        return soft_labels
    
    def _mixup(self, x, y):
        """Mixup数据增强"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # 混合标签
        y_soft = self._smooth_labels(y)
        y_soft_shuffled = y_soft[index]
        y_mixed = lam * y_soft + (1 - lam) * y_soft_shuffled
        
        return mixed_x, y_mixed
    
    def _cutmix(self, x, y):
        """CutMix数据增强"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # 生成随机边界框
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 执行剪切粘贴
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # 根据实际面积调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # 混合标签
        y_soft = self._smooth_labels(y)
        y_soft_shuffled = y_soft[index]
        y_mixed = lam * y_soft + (1 - lam) * y_soft_shuffled
        
        return mixed_x, y_mixed


def get_dataloaders(is_server: bool, batch_size: int, num_workers: int, 
                    use_mixup: bool = True,
                    use_randaugment: bool = True,
                    use_random_erasing: bool = True,
                    mixup_alpha: float = 0.75,
                    cutmix_alpha: float = 0.5,
                    label_smoothing: float = 0.1,
                    randaugment_config: str = 'rand-m9-n1-mstd0.4-inc1',
                    random_erasing_prob: float = 0.25,
                    data_root: str | Path | None = None,
                    **kwargs) -> tuple:
    """
    为CIFAR-100数据集创建并返回带高级数据增强的训练和测试DataLoader。

    Args:
        is_server (bool): 保留用于兼容旧 notebook，不再影响数据路径。
        batch_size (int): 批量大小。
        num_workers (int): 数据加载的工作进程数。
        use_mixup (bool): 是否使用Mixup/CutMix
        use_randaugment (bool): 是否使用RandAugment
        use_random_erasing (bool): 是否使用Random Erasing
        mixup_alpha (float): Mixup的Beta分布参数
        cutmix_alpha (float): CutMix的Beta分布参数
        label_smoothing (float): 标签平滑系数
        randaugment_config (str): RandAugment配置字符串
        random_erasing_prob (float): Random Erasing的概率
        data_root (str | Path | None): CIFAR-100 数据目录。默认使用仓库根目录下
            的 ``data/cifar-100``。
        **kwargs: 额外的关键字参数

    Returns:
        tuple: 一个包含 (trainloader, testloader, classes, is_dynamic_dataset, 
               mixup_fn, criterion) 的元组。
               - mixup_fn: Mixup/CutMix函数，在训练循环中使用
               - criterion: 损失函数（配合mixup使用SoftTargetCrossEntropy）
    """
    is_dynamic_dataset = False
    
    if data_root is None:
        data_dir = Path(__file__).resolve().parents[1] / 'data' / 'cifar-100'
    else:
        data_dir = Path(data_root).expanduser().resolve()

    # CIFAR-100 标准化参数
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # 构建训练数据增强pipeline
    train_transforms = []
    
    # 基础增强
    train_transforms.append(transforms.RandomCrop(32, padding=4))
    train_transforms.append(transforms.RandomHorizontalFlip())
    
    # RandAugment (在ToTensor之前应用)
    if use_randaugment and HAS_TIMM:
        # 注意：timm的rand_augment_transform期望PIL图像
        # 配置格式: rand-m{magnitude}-n{num_ops}-mstd{magnitude_std}
        ra_transform = rand_augment_transform(
            randaugment_config,
            hparams={
                'translate_const': 32,
                'img_mean': tuple([int(c * 255) for c in CIFAR100_TRAIN_MEAN])
            }
        )
        train_transforms.append(ra_transform)
    
    # ToTensor和Normalize
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD))
    
    # Random Erasing (在ToTensor之后应用)
    if use_random_erasing:
        train_transforms.append(transforms.RandomErasing(
            p=random_erasing_prob,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value=0,
            inplace=False
        ))
    
    transform_train = transforms.Compose(train_transforms)

    # 测试数据转换 (不使用数据增强)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD),
    ])

    # 创建数据集
    try:
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    except Exception as e:
        print(f"Error loading CIFAR-100 dataset from {data_dir}. Please check the path.")
        raise e

    # 创建数据加载器
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    # 创建Mixup/CutMix函数
    mixup_fn = None
    if use_mixup:
        if HAS_TIMM:
            # 使用timm的Mixup类
            mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=1.0,
                switch_prob=0.5,
                mode='batch',
                label_smoothing=label_smoothing,
                num_classes=100
            )
        else:
            # 使用自定义的MixupCutmix类
            mixup_fn = MixupCutmix(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=1.0,
                switch_prob=0.5,
                label_smoothing=label_smoothing,
                num_classes=100
            )
    
    # 创建损失函数
    if use_mixup:
        if HAS_TIMM:
            criterion = SoftTargetCrossEntropy()
        else:
            # 自定义软标签交叉熵
            criterion = lambda pred, target: torch.mean(
                torch.sum(-target * torch.log_softmax(pred, dim=-1), dim=-1)
            )
    elif label_smoothing > 0:
        if HAS_TIMM:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"CIFAR-100 (Data Augmented) 训练集大小: {len(trainset)}")
    print(f"CIFAR-100 测试集大小: {len(testset)}")
    print(f"数据增强配置:")
    print(f"  - RandAugment: {use_randaugment and HAS_TIMM} ({randaugment_config})")
    print(f"  - Mixup: {use_mixup} (alpha={mixup_alpha})")
    print(f"  - CutMix: {use_mixup} (alpha={cutmix_alpha})")
    print(f"  - Random Erasing: {use_random_erasing} (p={random_erasing_prob})")
    print(f"  - Label Smoothing: {label_smoothing}")
    
    classes = trainset.classes

    return trainloader, testloader, classes, is_dynamic_dataset, mixup_fn, criterion


def get_dataloaders_simple(is_server: bool, batch_size: int, num_workers: int, **kwargs) -> tuple:
    """
    简化版本的get_dataloaders，只返回4个元素，兼容原有接口。
    
    如果需要使用Mixup等高级功能，请使用get_dataloaders并在训练循环中手动应用。
    
    Args:
        is_server (bool): 判断数据集存放的根目录。
        batch_size (int): 批量大小。
        num_workers (int): 数据加载的工作进程数。
        **kwargs: 额外的关键字参数

    Returns:
        tuple: 一个包含 (trainloader, testloader, classes, is_dynamic_dataset) 的元组。
    """
    trainloader, testloader, classes, is_dynamic_dataset, _, _ = get_dataloaders(
        is_server=is_server, 
        batch_size=batch_size, 
        num_workers=num_workers,
        use_mixup=False,  # 简化版本不返回mixup_fn
        **kwargs
    )
    return trainloader, testloader, classes, is_dynamic_dataset


# 使用示例
if __name__ == "__main__":
    print("测试 CIFAR-100 数据增强版本数据加载器...")
    
    # 获取数据加载器
    trainloader, testloader, classes, is_dynamic, mixup_fn, criterion = get_dataloaders(
        is_server=False,
        batch_size=64,
        num_workers=0,  # Windows下设为0避免多进程问题
        use_mixup=True,
        use_randaugment=True,
        use_random_erasing=True
    )
    
    print(f"\n类别数量: {len(classes)}")
    print(f"训练批次数: {len(trainloader)}")
    print(f"测试批次数: {len(testloader)}")
    
    # 测试一个batch
    for images, labels in trainloader:
        print(f"\n原始batch:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        
        if mixup_fn is not None:
            # 应用Mixup/CutMix
            images_mixed, labels_mixed = mixup_fn(images, labels)
            print(f"\n混合后batch:")
            print(f"  图像形状: {images_mixed.shape}")
            print(f"  标签形状: {labels_mixed.shape}")
            print(f"  标签示例 (软标签): {labels_mixed[0, :5]}...")
        
        break
    
    print("\n测试完成!")
