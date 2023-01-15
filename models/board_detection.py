from torchvision.models import densenet201
import torch.nn as nn

class BoardDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.densenet = densenet201(pretrained=True)
        self.densenet.classifier = nn.Identity()

        # ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        # (a8, h8, h1, a1)
        self.out = nn.Linear(1920, 8) 
        
        self.act = nn.Sigmoid()
        print("Model initialized!")

    def forward(self, x):
        return self.act(self.out(self.densenet(x)))