from torch import nn
from .memory import MemoryModule
from .decoder import Decoder
from .msff import MSFF
from typing import List, Tuple


class DMemSeg(nn.Module):
    def __init__(self, feature_extractor, num_memory: int, feature_shapes: List[Tuple[int, int, int]],
                 threshold: float = 0.5, epsilon: float = 1e-12):
        super(DMemSeg, self).__init__()

        self.memory_module = MemoryModule(num_memory, feature_shapes, threshold, epsilon)
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

    def forward(self, inputs):
        # extract features
        features = self.feature_extractor(inputs)
        feature0 = features[0]
        feature13 = features[1:-1]
        feature4 = features[-1]

        # extract concatenated information(CI)
        concat_features, weight = self.memory_module(feature13)
        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features=concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output=feature4,
            concat_features=[feature0] + msff_outputs
        )

        return predicted_mask, weight
