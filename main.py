# installed imports
import torch

# local imports
from utils import *
from dataloader import BoardDetectorDataset, PieceDetectorDataset
from trainers import train_piece_detector, train_board_detector
from models import PieceDetector, BoardDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    # epochs = 25
    # batch_size = 2
    # lr = 3e-4
    # weight_decay = 0.0005
    # train_piece_detector(batch_size=batch_size,
    #                      epochs=epochs,
    #                      weights_save_folder='./models/checkpoints/piece_detector/2',
    #                      weights_load_path='./models/checkpoints/piece_detector/2/weight', 
    #                      learning_rate=lr,
    #                      weight_decay=weight_decay,
    #                      from_pretrained=False,
    #                      mixed_precision_training=True,
    #                      device=device)
    # print('Training Complete!')
    # show_piece_detector_results(idx=5,
    #                             weights_path='models/checkpoints/piece_detector/2/weight', 
    #                             json_file='dataloader/data/piece_data/train/_annotations.coco.json',
    #                             device=device)

    # epochs = 50
    # batch_size = 64
    # lr = 3e-4
    # train_board_detector(model='resnet',
    #                      epochs=epochs,
    #                      batch_size=batch_size,
    #                      learning_rate=lr,
    #                      mixed_precision_training=True,
    #                      weights_load_path=None,
    #                      weights_save_folder='models/checkpoints/board_detector/resnet',
    #                      device=device)
    # print('Training Complete!')

    weights = 'models/checkpoints/board_detector/resnet/weight'
    json_file = 'dataloader/data/board_data/train/_annotations.coco.json'

    board_detector = BoardDetector(model='resnet').to(device)
    board_detector.load_state_dict(torch.load(weights))
    board_detector.eval()
    board_data = BoardDetectorDataset(json_file)

    idx = 0
    while True:
        try:
            idx = int(input('idx:'))
        except:
            quit()

        show_board_detector_results(idx, board_detector, board_data, device)

if __name__ == '__main__':
    main()