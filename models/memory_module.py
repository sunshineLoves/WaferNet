import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from typing import List, Callable, Tuple

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
        weight = F.normalize(weight, p=1, dim=1)
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
        self.memory_units = nn.ModuleList([MemoryUnit(memory_size, shape, self.hard_shrink) for shape in shapes])

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
                        self.memory_information[f'level{i}'] = torch.cat([self.memory_information[f'level{i}'], features_l], dim=0)

                        
    def _calc_diff(self, features: List[torch.Tensor]) -> torch.Tensor:
        # batch size X the number of samples saved in memory
        diff_bank = torch.zeros(features[0].size(0), self.nb_memory_sample).to(self.device)

        # level
        for l, level in enumerate(self.memory_information.keys()):
            # batch
            for b_idx, features_b in enumerate(features[l]):
                # calculate l2 loss
                diff = F.mse_loss(
                    input     = torch.repeat_interleave(features_b.unsqueeze(0), repeats=self.nb_memory_sample, dim=0), 
                    target    = self.memory_information[level], 
                    reduction ='none'
                ).mean(dim=[1,2,3])

                # sum loss
                diff_bank[b_idx] += diff
                
        return diff_bank
        
    
    def select(self, features: List[torch.Tensor]) -> torch.Tensor:
        # calculate difference between features and normal features of memory bank
        diff_bank = self._calc_diff(features=features)
        
        # concatenate features with minimum difference features of memory bank
        for l, level in enumerate(self.memory_information.keys()):
            
            selected_features = torch.index_select(self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            diff_features = F.mse_loss(selected_features, features[l], reduction='none')
            features[l] = torch.cat([features[l], diff_features], dim=1)
            
        return features
    