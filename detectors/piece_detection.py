# installed imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn as mobile_fasterrcnn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large as mobile_ssd
import torch.nn as nn

class PieceDetector(nn.Module):
    """
    Just a FasterrRCNN with Resnet50 backbone
    """
    def __init__(self, model='faster_rcnn', pretrained=True):
        super().__init__()
        max_objs = 32 # total of 32 pieces on board at any point
        num_classes = 12 + 1 # 12 types of pieces (white pawn, black rook ... etc) + 1 for no obj
        fasterrcnn_model = fasterrcnn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None, 
                                      box_detections_per_img=max_objs)
        in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
        fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

        self.detector = fasterrcnn_model

    def forward(self, *args):
        return self.detector(*args)