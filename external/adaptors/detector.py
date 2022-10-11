"""Generic detector."""
import torch

from yolox import get_model


class Detector(torch.nn.Module):
    def __init__(self, model_type, weight_path):
        super().__init__()

    def forward(self):
        pass
