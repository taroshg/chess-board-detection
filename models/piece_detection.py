from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
import torch.nn as nn

class PieceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = fasterrcnn(pretrained=True)

    def forward(self, *args):
        return self.detector(*args)