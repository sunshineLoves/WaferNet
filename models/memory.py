import math

import torch
from torch import nn
from typing import List, Tuple
from torch.nn import functional as F
import numpy as np


class MemoryModule(nn.Module):
    def __init__(self, memory_size: int, shapes: List[Tuple[int, int, int]], threshold: float = 0.5,
                 epsilon: float = 1e-12):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.num_feature = len(shapes)
        self.shapes = shapes
        self.memory_list = nn.ParameterList(
            [nn.Parameter(torch.Tensor(self.memory_size, C * H * W)) for C, H, W in shapes])
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self):
        for memory in self.memory_list:
            stdv = 1. / math.sqrt(memory.size(1))
            memory.data.uniform_(-stdv, stdv)

    def hard_shrink(self, weight: torch.Tensor):
        output = (F.relu(weight - self.threshold) * weight) / (torch.abs(weight - self.threshold) + self.epsilon)
        return output

    def forward(self, features: List[torch.Tensor]):
        batch_size = features[0].size(0)
        memory_weights_featured = []
        for i, memory in enumerate(self.memory_list):
            # (batch_size, C * H * W)
            flatten_feature = features[i].view(batch_size, -1)
            # (memory_size, C * H * W)
            normalized_feature = F.normalize(flatten_feature, p=2, dim=1)
            normalized_memory = F.normalize(memory, p=2, dim=1)
            # (batch_size, memory_size)
            weight = F.linear(normalized_feature, normalized_memory)
            memory_weights_featured += [weight]
        # (batch_size, memory_size)
        memory_weights_featured = torch.stack(memory_weights_featured)
        memory_weight = torch.mean(memory_weights_featured, dim=0)
        memory_weight = F.softmax(memory_weight, dim=1)
        print(memory_weight)
        memory_weight = self.hard_shrink(memory_weight)
        print(memory_weight)
        memory_weight = F.normalize(memory_weight, p=2, dim=1)
        print(memory_weight)

        attention_features = []
        for feature, memory, shape in zip(features, self.memory_list, self.shapes):
            # memory_weight : (batch_size, memory_size)
            # memory        : (memory_size, C * H * W)
            memory_feature = torch.matmul(memory_weight, memory)
            memory_feature = memory_feature.view(batch_size, *shape)
            attention_features += [torch.cat((feature, memory_feature), dim=1)]

        return attention_features, memory_weight


class MemoryBank:
    def __init__(self, normal_dataset, nb_memory_sample: int = 30, device='cpu'):
        self.device = device

        # memory bank
        self.memory_information = {}

        # normal dataset
        self.normal_dataset = normal_dataset

        # the number of samples saved in memory bank
        self.nb_memory_sample = nb_memory_sample

    def update(self, feature_extractor):
        feature_extractor.eval()

        # define sample index
        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        # extract features and save features into memory bank
        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                # select image
                input_normal, _, _ = self.normal_dataset[samples_idx[i]]
                input_normal = input_normal.to(self.device)

                # extract features
                features = feature_extractor(input_normal.unsqueeze(0))

                # save features into memoery bank
                for i, features_l in enumerate(features[1:-1]):
                    if f'level{i}' not in self.memory_information.keys():
                        self.memory_information[f'level{i}'] = features_l
                    else:
                        self.memory_information[f'level{i}'] = torch.cat(
                            [self.memory_information[f'level{i}'], features_l], dim=0)

    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l, level in enumerate(self.memory_information.keys()):
            # batch
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss
                diff = F.mse_loss(
                    input=torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0),
                    target=self.memory_information[level],
                    reduction='none'
                ).mean(dim=[1, 2, 3])

                # sum loss
                diff_bank[b_idx] += diff

        return diff_bank

    def select(self, features: List[torch.Tensor]):
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features=features)

        # concatenate features with minimum difference features of memory bank
        for l, level in enumerate(self.memory_information.keys()):
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features
