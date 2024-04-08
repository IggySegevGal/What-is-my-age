import torch
import torch.nn as nn

# Load the pre-trained DINO model

class DinoWithFC(nn.Module):
    def __init__(self, num_classes):
        super(DinoWithFC, self).__init__()
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.transformer = dinov2_vits14
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
