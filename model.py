import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=143):
        super(EfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)


    def forward(self, x):
            return self.model(x)
