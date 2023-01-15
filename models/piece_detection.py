from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn

class PieceDetector():
    def __init__(self):
        super().__init__()
        self.detector = fasterrcnn(pretrained=True)

    def forward(self, x):
        return self.detector