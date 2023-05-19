# installed imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn as mobile_fasterrcnn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large as mobile_ssd
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models import resnet50, ResNet50_Weights
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


class PiecePointDetection(nn.Module):
    def __init__(self):
        def __init__(self, pretrained=True, model='squeezenet', target='points'):
            super().__init__()
            self.target = target
            assert (model == 'densenet' or
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
