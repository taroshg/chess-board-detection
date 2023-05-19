# installed imports
import torch
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn


class BoardDetector(nn.Module):
    """
    Simple pretrained CNN with classifer output of (1, 8) 
    for (x, y, x, y, x, y, x, y) keypoints of board corners
    """
    def __init__(self, pretrained=True, model='squeezenet', target='points'):
        super().__init__()
        self.target = target
        assert(model == 'densenet' or 
               model == 'squeezenet' or
               model == 'resnet'), "incorrect model name!"
        if model == 'squeezenet':
            self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT if pretrained else None).features
            self.out = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 8)
            )
        elif model == 'densenet':
            self.model = densenet201(weights=DenseNet201_Weights.DEFAULT if pretrained else None).features
            self.out = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(1920, 8)
            )
        elif model == 'resnet':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.model.fc = nn.Identity()
            self.out = nn.Linear(2048, 8)
        
        self.act = nn.Sigmoid()
        print("Model initialized!")

    def forward(self, x):
        x = self.model(x)
        x = self.out(x)

        return self.act(x)

    def loss_function(self):
        if self.target == 'points':
            return nn.MSELoss()
        if self.target == 'mask':
            return nn.BCELoss()


class BoardMask(nn.Module):
    def __init__(self, pretrained=False):
        mask = maskrcnn(weights=MaskRCNN_ResNet50_FPN_Weights if pretrained else None, num_classes=1)
