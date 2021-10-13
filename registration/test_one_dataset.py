import argparse
import logging
import os
import sys

import numpy as np
from numpy.random.mtrand import sample
import torch
import torchvision
from tqdm import tqdm

from data.modelnet_loader_torch import ModelNetCls
from models import pcrnet
from src import ChamferDistance, FPSSampler, RandomSampler, SampleNet
from src import sputils
from src.pctransforms import OnUnitCube, PointcloudToTensor
from src.qdataset import QuaternionFixedDataset, QuaternionTransform, rad_to_deg


sampler = SampleNet(
    num_out_points=64,
    bottleneck_size=128,
    group_size=8,  # 8 or 10
    initial_temperature=1.0,
    input_shape="bnc",
    output_shape="bnc",
    skip_projection=False,
)

model_path = "log/SAMPLENET64_model_best.pth"
sampler.load_state_dict(torch.load(model_path))
