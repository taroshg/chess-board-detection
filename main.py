# installed imports
import torch

# local imports
from utils import *
from dataloader import BoardDetectorDataset, PieceDetectorDataset
from trainers import train_piece_detector, train_board_detector
from models import PieceDetector, BoardDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    # # epochs = 25
    # # batch_size = 2
    # # lr = 3e-4
    # # weight_decay = 0.0005
    # # train_piece_detector(batch_size=batch_size,
    # #                      epochs=epochs,
    # #                      weights_save_folder='./models/checkpoints/piece_detector/2',
    # #                      weights_load_path='./models/checkpoints/piece_detector/2/weight', 
    # #                      learning_rate=lr,
    # #                      weight_decay=weight_decay,
    # #                      from_pretrained=False,
    # #                      mixed_precision_training=True,
    # #                      device=device)
    # # print('Training Complete!')
    # show_piece_detector_results(idx=5,
    #                             weights_path='models/checkpoints/piece_detector/2/weight', 
    #                             json_file='dataloader/data/piece_data/train/_annotations.coco.json',
    #                             device=device)

    # epochs = 75
    # train_board_detector(epochs=epochs,
    #                      weights_load_path='models/checkpoints/board_detector/6/weight',
    #                      weights_save_folder='models/checkpoints/board_detector/6')
    # print('Training Complete!')
    
    weights_path='models/checkpoints/board_detector/6/weight'
    json_file='dataloader/data/board_data/test/_annotations.coco.json'
    board_detector = BoardDetector().to(device)
    board_detector.load_state_dict(torch.load(weights_path))
    board_detector.eval()

    board_data = BoardDetectorDataset(json_file)
    while True:
        print("idx:")
        try:
            idx = int(input())
        except:
            quit()
        show_board_detector_results(idx,
                                    board_detector,
                                    board_data,
                                    device=device)
    
if __name__ == '__main__':
    main()