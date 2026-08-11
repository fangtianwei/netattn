import unittest

import torch

from models.ResNet_NetAttn_CIFAR100 import (
    ResNet_NetAttn_CIFAR100,
    subnet_bits_to_index,
    subnet_index_to_bits,
)


class DistinctBranchBlock:
    def forward_branches(self, x):
        return x, x + 1


class FullNetAttnPathOrderTests(unittest.TestCase):
    def test_path_bit_round_trip(self):
        for index in range(256):
            with self.subTest(index=index):
                bits = subnet_index_to_bits(index, 8)
                self.assertEqual(len(bits), 8)
                self.assertEqual(subnet_bits_to_index(bits), index)

    def test_known_path_mappings(self):
        self.assertEqual(subnet_index_to_bits(1, 8), "10000000")
        self.assertEqual(subnet_index_to_bits(63, 8), "11111100")
        self.assertEqual(subnet_index_to_bits(128, 8), "00000001")

    def test_internal_index_bits_follow_shallow_to_deep_blocks(self):
        model = object.__new__(ResNet_NetAttn_CIFAR100)
        paths = torch.zeros(1, 1, 1, 1, 1)
        block = DistinctBranchBlock()
        for block_position in range(3):
            paths = model._expand_subnetworks(block, paths)
            for index, value in enumerate(paths[:, 0, 0, 0, 0].tolist()):
                bits = subnet_index_to_bits(index, block_position + 1)
                self.assertEqual(value, bits.count("1"))


if __name__ == "__main__":
    unittest.main()
