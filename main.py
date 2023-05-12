# installed imports
import torchmetrics.detection.mean_ap
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
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
    print('running cuda...')
    device = "cuda"
elif torch.backends.mps.is_available():
    print('running mps...')
    device = "mps"

def main():
    """
    PIECE DETECTOR TRAINING SECTION
    """
    img_size = 320
    MODEL = f'{img_size}_faster_rcnn'
    json_file = f'dataloader/data/piece_data/piece_detection_coco_{img_size}.json'
    PRETRAINED = True

    # epochs = 10
    # batch_size = [2]
    # lr = [1e-4]
    # weight_decay = 0.0005
    # writer = SummaryWriter(f'detectors/checkpoints/piece_detector/{MODEL}/tensorboard')
    #
    # m_ap = None
    # for b in batch_size:
    #     for l in lr:
    #         loss, lowest_loss, m_ap = train_piece_detector(batch_size=b,  # hyperparameter search (batch_size)
    #                                                  epochs=epochs,
    #                                                  weights_load_path=f'detectors/checkpoints/piece_detector/{MODEL}/weight',
    #                                                  weights_save_folder=f'./detectors/checkpoints/piece_detector/{MODEL}',
    #                                                  json_file=json_file,
    #                                                  img_size=(img_size, img_size),
    #                                                  learning_rate=l,  # hyperparameter search (learning_rate)
    #                                                  weight_decay=weight_decay,
    #                                                  from_pretrained=PRETRAINED,
    #                                                  mixed_precision_training=torch.cuda.is_available(),
    #                                                  writer=writer,  # optional summary writer added!
    #                                                  step=1700,
    #                                                  device=device)
    #
    #         writer.add_hparams({'learning rate': l, 'batch size': b},
    #                            {'lowest loss': lowest_loss, 'last loss': loss, 'mAP': m_ap['map']})
    # writer.close()
    # print(m_ap)
    # print('Training Complete!')


    """
    PIECE DETECTOR VIEW OUTPUT
    """
    weights = f'detectors/checkpoints/piece_detector/{MODEL}/weight'
    piece_detector = PieceDetector(pretrained=PRETRAINED).to(device)
    piece_detector.load_state_dict(torch.load(weights, map_location=device))
    piece_detector.eval()
    piece_data = PieceDetectorDataset(json_file, size=(img_size, img_size))

    # img, target = piece_data[0]
    # targets = [{k: v.to(device) for k, v in target.items()}]
    # preds = piece_detector(img.unsqueeze(0).to(device))
    # mAP = torchmetrics.detection.mean_ap.MeanAveragePrecision()
    # mAP.update(preds, targets)
    # out = mAP.compute()
    # print(out)

    idx = 0
    while idx < len(piece_data):
        show_piece_detector_results(idx, piece_detector, piece_data, device)
        try:
            idx += 1
        except:
            quit()

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
    """
    BOARD DETECTOR VIEW OUTPUT
    """
    # weights = f'detectors/checkpoints/board_detector/{MODEL}/weight'
    # json_file = 'dataloader/data/board_data/board_dataset_coco.json'
    #
    # board_detector = BoardDetector(model=MODEL).to(device)
    # board_detector.load_state_dict(torch.load(weights, map_location=device))
    # board_detector.eval()
    # board_data = BoardDetectorDataset(json_file, size=320, target="mask")
    # for i in range(len(board_data)):
    #     img, mask = board_data[i]
    #     f, ax = plt.subplots(1, 2, figsize=(12, 10))
    #     ax[0].imshow(mask[0])
    #     ax[1].imshow(img.permute(1, 2, 0))
    #     plt.show()
    """
    BOARD DETECTOR SAVE OUTPUT
    """
    # for idx in range(len(board_data)):
    #     show_board_detector_results_masked(idx, board_detector, board_data, device)
    # json_file = 'dataloader/data/board_data/board_dataset_coco.json'
    # board_data = BoardDetectorDataset(json_file)
    # generate_warped_board_images(board_data=board_data, 
    #                              load_folder='dataloader/data/raw', 
    #                              save_folder='dataloader/data/raw_warped', 
    #                              size=(1000, 1000),
    #                              device=device)


if __name__ == '__main__':
    main()
