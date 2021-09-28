from chamfer_distance.chamfer_distance import ChamferDistance
import itertools
import numpy as np
import copy
import torch

target = itertools.product("10", repeat=3)
target = np.asarray(list(target))
target = target.astype(np.float32)
print(target)

source = copy.deepcopy(target)
for i in range(len(source)):
    if np.all(source[i] == np.array([1.0, 0.0, 0.0])):
        source[i] = np.array([2.0, 0.0, 0.0])
print(source)


target = torch.from_numpy(target).clone().detach().cuda()
source = torch.from_numpy(source).clone().detach().cuda()

target = target.permute(1, 0)
source = source.permute(1, 0)

print(target.shape)
print(source.shape)

target = torch.unsqueeze(target, 0).clone().detach().cuda()
source = torch.unsqueeze(source, 0).clone().detach().cuda()

cost_p1_p2, cost_p2_p1 = ChamferDistance()(target, source)
print(cost_p1_p2)
print(cost_p2_p1)
