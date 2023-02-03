# installed imports
import torch

# local imports
from utils import *
from dataloader import BoardDetectorDataset, PieceDetectorDataset
from trainers import train_piece_detector, train_board_detector
from models import PieceDetector, BoardDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    MODEL = '0'
    # epochs = 25
    # batch_size = 2
    # lr = 3e-4
    # weight_decay = 0.0005
    # train_piece_detector(batch_size=batch_size,
    #                      epochs=epochs,
    #                      weights_save_folder='./models/checkpoints/piece_detector/0',
    #                      weights_load_path=None, 
    #                      learning_rate=lr,
    #                      weight_decay=weight_decay,
    #                      from_pretrained=True,
    #                      mixed_precision_training=True,
    #                      device=device)
    # print('Training Complete!')
    weights = f'models/checkpoints/piece_detector/{MODEL}/weight'
    json_file = 'dataloader/data/piece_data/train/_annotations.coco.json'

    piece_detector = PieceDetector().to(device)
    piece_detector.load_state_dict(torch.load(weights, map_location=device))
    piece_detector.eval()
    piece_data = PieceDetectorDataset(json_file)

    idx = 0
    while idx < len(piece_data):
        print(idx)
        show_piece_detector_results(idx, piece_detector, piece_data, device)
        try:
            idx += 1
        except:
            quit()


    # MODEL = 'squeezenet'
    # epochs = 100
    # batch_size = 64
    # lr = 3e-4
    # train_board_detector(model=MODEL,
    #                      epochs=epochs,
    #                      batch_size=batch_size,
    #                      learning_rate=lr,
    #                      mixed_precision_training=True if device=='cuda' else False,
    #                      weights_load_path=None,
    #                      weights_save_folder=f'models/checkpoints/board_detector/{MODEL}',
    #                      device=device)
    # print('Training Complete!')

    # weights = f'models/checkpoints/board_detector/{MODEL}/weight'
    # json_file = 'dataloader/data/board_data/train/_annotations.coco.json'

    # board_detector = BoardDetector(model=MODEL).to(device)
    # board_detector.load_state_dict(torch.load(weights, map_location=device))
    # board_detector.eval()
    # board_data = BoardDetectorDataset(json_file)

    # idx = 0
    # while idx < len(board_data):
    #     print(idx)
    #     show_board_detector_results(idx, board_detector, board_data, device)
    #     try:
    #         idx += 1
    #     except:
    #         quit()

if __name__ == '__main__':
    main()