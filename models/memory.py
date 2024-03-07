import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Callable
from torch.nn import functional as F


class MemoryUnit(nn.Module):
    def __init__(self, memory_size: int, shape: Tuple[int, int, int],
                 hard_shrink: Callable[[torch.Tensor], torch.Tensor]):
        super(MemoryUnit, self).__init__()
        self.memory_size = memory_size
        self.shape = shape
        self.hard_shrink = hard_shrink
        # (memory_size, C * H * W)
        self.memory = nn.Parameter(torch.Tensor(self.memory_size, np.prod(shape)))

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = feature.size(0)
        # (batch_size, C * H * W)
        flatten_feature = feature.view(batch_size, -1)
        normalized_feature = F.normalize(flatten_feature, p=2, dim=1)
        # (memory_size, C * H * W)
        normalized_memory = F.normalize(self.memory, p=2, dim=1)
        # (batch_size, memory_size)
        weight = F.linear(normalized_feature, normalized_memory)
        weight = F.softmax(weight, dim=1)
        weight = self.hard_shrink(weight)
        weight = F.softmax(weight, dim=1)
        memory_feature = torch.matmul(weight, self.memory)
        memory_feature = memory_feature.view(batch_size, *self.shape)
        return memory_feature, weight


class MemoryModule(nn.Module):
    def __init__(self, memory_size: int, shapes: List[Tuple[int, int, int]], threshold: float = 0.0025,
                 epsilon: float = 1e-12):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.num_feature = len(shapes)
        self.shapes = shapes
        self.threshold = threshold
        self.epsilon = epsilon
        self.memory_units = [MemoryUnit(memory_size, shape, self.hard_shrink) for shape in shapes]

    def hard_shrink(self, weight: torch.Tensor) -> torch.Tensor:
        output = (F.relu(weight - self.threshold) * weight) / (torch.abs(weight - self.threshold) + self.epsilon)
        return output

    def forward(self, features: List[torch.Tensor]):
        concat_features = []
        memory_weights = []

        for memory_unit, feature in zip(self.memory_units, features):
            memory_feature, weight = memory_unit(feature)
            concat_feature = torch.cat((memory_feature, feature), dim=1)
            concat_features += [concat_feature]
            memory_weights += [weight]

        return concat_features, memory_weights
