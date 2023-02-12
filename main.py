# installed imports
import torch

# default imports
from datetime import datetime as time

# local imports
from dataloader import BoardDetectorDataset, PieceDetectorDataset
from detectors import BoardDetector, PieceDetector
from trainers import train_board_detector, train_piece_detector
from helpers import *

device = "cpu"
if torch.cuda.is_available():
    print('running mps...')
    device = "cuda"
elif torch.backends.mps.is_available():
    print('running mps...')
    device = "mps"


def main():
    """
    PIECE DETECTOR TRAINING SECTION
    """
    # MODEL = 'faster_rcnn_M1'
    # PRETRAINED = False
    # epochs = 100
    # batch_size = 32
    # lr = 3e-4
    # weight_decay = 0.0005
    # train_piece_detector(batch_size=batch_size,
    #                      epochs=epochs,
    #                      weights_load_path=None,
    #                      weights_save_folder=f'./detections/checkpoints/piece_detector/{MODEL}',
    #                      learning_rate=lr,
    #                      weight_decay=weight_decay,
    #                      from_pretrained=PRETRAINED,
    #                      mixed_precision_training=torch.cuda.is_available(),
    #                      device=device)
    # print('Training Complete!')
    #
    # weights = f'detections/checkpoints/piece_detector/{MODEL}/weight'
    # json_file = 'dataloader/data/piece_data/piece_detection_coco_1000.json'
    # piece_detector = PieceDetector(pretrained=PRETRAINED).to(device)
    # piece_detector.load_state_dict(torch.load(weights, map_location=device))
    # piece_detector.eval()
    # piece_data = PieceDetectorDataset(json_file)
    #
    # idx = 0
    # while idx < len(piece_data):
    #     show_piece_detector_results(idx, piece_detector, piece_data, device)
    #     try:
    #         idx += 1
    #     finally:
    #         quit()

    """
    BOARD DETECTOR TRAINING SECTION
    """
    # MODEL = 'squeezenet'
    # epochs = 100
    # batch_size = 64
    # lr = 3e-4
    # train_board_detector(model=MODEL,
    #                      epochs=epochs,
    #                      batch_size=batch_size,
    #                      learning_rate=lr,
    #                      mixed_precision_training=torch.cuda.is_available(),
    #                      weights_load_path=None,
    #                      weights_save_folder=f'detectors/checkpoints/board_detector/{MODEL}',
    #                      device=device)
    # print('Training Complete!')

    # weights = f'detectors/checkpoints/board_detector/{MODEL}/weight'
    # json_file = 'dataloader/data/board_data/board_dataset_coco.json'

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

    # json_file = 'dataloader/data/board_data/board_dataset_coco.json'
    # board_data = BoardDetectorDataset(json_file)
    # generate_warped_board_images(board_data=board_data, 
    #                              load_folder='dataloader/data/raw', 
    #                              save_folder='dataloader/data/raw_warped', 
    #                              size=(1000, 1000),
    #                              device=device)


if __name__ == '__main__':
    main()
