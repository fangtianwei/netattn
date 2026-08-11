"""Small CPU forward checks for the three released ANN model families."""

import torch

from models.ResNet_CIFAR100 import BasicBlock as BaselineBlock
from models.ResNet_CIFAR100 import ResNet_CIFAR100
from models.ResNet_NetAttn_CIFAR100 import BasicBlock as FullBlock
from models.ResNet_NetAttn_CIFAR100 import ResNet_NetAttn_CIFAR100
from models.ResNet_NetAttn_Path9_CIFAR100 import BasicBlock as PathBlock
from models.ResNet_NetAttn_Path9_CIFAR100 import ResNet_NetAttn_Path9_CIFAR100


def main() -> None:
    torch.manual_seed(42)
    inputs = torch.randn(1, 3, 32, 32)
    models = {
        "baseline": ResNet_CIFAR100(BaselineBlock, [1, 1, 1, 1]),
        "full_netattn": ResNet_NetAttn_CIFAR100(FullBlock, [1, 1, 1, 1]),
        "path9": ResNet_NetAttn_Path9_CIFAR100(PathBlock, [2, 2, 2, 2]),
    }
    with torch.inference_mode():
        for name, model in models.items():
            model.eval()
            output = model(inputs)
            assert output.shape == (1, 100), (name, output.shape)
            assert torch.isfinite(output).all(), name
            print(f"{name}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
