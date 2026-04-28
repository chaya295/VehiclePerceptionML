import torch
import torch.nn as nn
import torchvision.models as models

class VehiclePerceptionModel(nn.Module):
    def __init__(self, num_classes):
        super(VehiclePerceptionModel, self).__init__()
        # Load the pre-trained ResNet18 model
        self.base_model = models.resnet18(pretrained=True)
        # Modify the final fully connected layer
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
