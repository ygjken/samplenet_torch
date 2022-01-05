from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import numpy as np


class TransformsForTwo:
    def __init__(self) -> None:
        pass

    def __call__(self, source, target):
        src, tgt = PointcloudToTensorForTwo(source, target)
        src, tgt = OnUnitCubeForTwo(src, tgt)
        return src, tgt


class PointcloudToTensorForTwo:
    def __init__(self) -> None:
        pass

    def __call__(self, source, target):
        return torch.from_numpy(source).float(), torch.from_numpy(target).float()


class OnUnitCubeForTwo:
    def __init__(self) -> None:
        pass

    def method(self, source, target):
        all_points = torch.cat((source, target), 0)

        c = torch.max(all_points, dim=0)[0] - torch.min(all_points, dim=0)[0]
        s = torch.max(c)

        src = source / s
        tgt = target / s

        return src - src.mean(dim=0, keepdim=True), tgt - tgt.mean(dim=0, keepdim=True)

    def __call__(self, source, target):
        return self.method(source, target)
