import torch
import torch.nn as nn
import torchvision.models as models

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        # Load a pre-trained backbone model (e.g., ResNet50)
        self.backbone = models.resnet50(pretrained=True)

        # Feature pyramid layers
        self.layer1 = nn.Conv2d(256, 256, kernel_size=1)
        self.layer2 = nn.Conv2d(512, 256, kernel_size=1)
        self.layer3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.layer4 = nn.Conv2d(2048, 256, kernel_size=1)

    def forward(self, x):
        # Backbone feature extraction
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        # FPN feature fusion
        p4 = self.layer1(c4)
        p3 = self.layer2(c3)
        p2 = self.layer3(c2)
        p1 = self.layer4(c1)

        return p4, p3, p2, p1

# Create an instance of the FPN model
fpn = FPN()

# Test the FPN model with random input
input = torch.randn(1, 3, 256, 256)
output = fpn(input)
