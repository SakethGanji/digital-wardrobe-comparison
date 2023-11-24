import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes_dict):
        super(EfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.base_model = models.efficientnet_b0(weights=weights)
        num_ftrs = self.base_model.classifier[1].in_features

        self.classifiers = nn.ModuleDict({
            col: nn.Linear(num_ftrs, num_classes)
            for col, num_classes in num_classes_dict.items()
        })

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        return {col: classifier(x) for col, classifier in self.classifiers.items()}
