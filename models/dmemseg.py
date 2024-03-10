import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF
from .memory_module import MemoryModule
from typing import List, Tuple

class DMemSeg(nn.Module):
    def __init__(self, feature_extractor, memory_size: int, feature_channels: List[int],
                 threshold: float = 0.0025, epsilon: float = 1e-12):
        super(DMemSeg, self).__init__()

        self.memory_module = MemoryModule(memory_size, feature_channels, threshold, epsilon)
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        # extract features
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features, weights = self.memory_module(f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + msff_outputs
        )

        return predicted_mask, weights
