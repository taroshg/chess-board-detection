# installed imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn

class PieceDetector(nn.Module):
    """
    Just a FasterrRCNN with Resnet50 backbone
    """
    def __init__(self):
        super().__init__()
        self.in_features = 32
        fasterrcnn_model = fasterrcnn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
        fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=self.in_features)

        self.detector = fasterrcnn_model

    def forward(self, *args):
        return self.detector(*args)