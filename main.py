# installed imports
import torchmetrics.detection.mean_ap
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch

# default imports
from datetime import datetime as time

# local imports
from dataloader import BoardDetectorDataset, PieceDetectorDataset, PieceDetectorCOGDataset
from models import BoardDetector, PieceDetector
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
    json_file = 'dataloader/data/data_generation/images/piece_coco.json'
    root_folder = 'dataloader/data/data_generation/images'
    PRETRAINED = True

    epochs = 10
    batch_size = [8]
    lr = [1e-4]
    weight_decay = 0.0005
    writer = SummaryWriter(f'models/checkpoints/piece_detector/{MODEL}/tensorboard_synthetic')
    for b in batch_size:
        for l in lr:
            loss, lowest_loss = train_piece_detector(batch_size=b,  # hyperparameter search (batch_size)
                                                     epochs=epochs,
                                                     weights_load_path='models/checkpoints/piece_detector/320_faster_rcnn/weight',
                                                     weights_save_folder='models/checkpoints/piece_detector/320_faster_rcnn/',
                                                     weights_name='synthetic_weight',
                                                     json_file=json_file,
                                                     root_folder=root_folder,
                                                     img_size=(img_size, img_size),
                                                     learning_rate=l,  # hyperparameter search (learning_rate)
                                                     weight_decay=weight_decay,
                                                     from_pretrained=PRETRAINED,
                                                     mixed_precision_training=torch.cuda.is_available(),
                                                     writer=writer,  # optional summary writer added!
                                                     step=0,
                                                     device=device)

            # writer.add_hparams({'learning rate': l, 'batch size': b},
            #                    {'lowest loss': lowest_loss, 'last loss': loss, 'mAP': m_ap['map']})
    writer.close()
    print('Training Complete!')

    """
    PIECE DETECTOR VIEW OUTPUT
    """
    # weights = f'models/checkpoints/piece_detector/{MODEL}/weight'
    # piece_detector = PieceDetector(pretrained=PRETRAINED).to(device)
    # piece_detector.load_state_dict(torch.load(weights, map_location=device))
    # piece_detector.eval()
    # piece_data = PieceDetectorDataset(root=root_folder, json_file=json_file, size=(img_size, img_size))
    #
    # img, _ = piece_data[4]
    # features = piece_detector.detector.backbone(img.unsqueeze(0).to(device))
    # save_location = 'dataloader/data/trained_fpn_output/80x80'
    # for i, feature in tqdm(enumerate(features['0'][0])):
    #     plt.imshow(feature.detach().numpy())
    #     plt.axis('off')
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     plt.savefig(f'{save_location}/{i}.jpg', bbox_inches='tight', pad_inches=0)

    # f, ax = plt.subplots(1, 5, figsize=(12, 5))
    # ax[0].imshow(features['0'][0][0].detach().numpy())
    # ax[1].imshow(features['1'][0][0].detach().numpy())
    # ax[2].imshow(features['2'][0][0].detach().numpy())
    # ax[3].imshow(features['3'][0][0].detach().numpy())
    # ax[4].imshow(features['pool'][0][0].detach().numpy())
    # plt.show()

    # img, target = piece_data[0]
    # targets = [{k: v.to(device) for k, v in target.items()}]
    # preds = piece_detector(img.unsqueeze(0).to(device))
    # mAP = torchmetrics.detection.mean_ap.MeanAveragePrecision()
    # mAP.update(preds, targets)
    # out = mAP.compute()
    # print(out)

    # idx = 0
    # while idx < len(piece_data):
    #     show_piece_detector_results(idx, piece_detector, piece_data, device)
    #     try:
    #         idx += 1
    #     except:
    #         quit()

    """
    BOARD DETECTOR TRAINING SECTION
    """
    # img_size = 320
    # MODEL = 'resnet'
    # json_file = 'dataloader/data/board_detector_coco.json'
    # root_folder = 'dataloader/data/raw/'
    # PRETRAINED = True

    # epochs = 10
    # batch_size = [16]
    # lr = [1e-4]
    #
    # writer = SummaryWriter(f'models/checkpoints/board_detector/{MODEL}/tensorboard')
    # step=10
    # last_loss = 0
    # lowest_loss = 0
    # for b in batch_size:
    #     for l in lr:
    #         print(f'currently training for (batch_size: {b}, lr: {l})')
    #         last_loss, lowest_loss = train_board_detector(model=MODEL,
    #                                                       epochs=epochs,
    #                                                       batch_size=b,
    #                                                       learning_rate=l,
    #                                                       mixed_precision_training=torch.cuda.is_available(),
    #                                                       weights_load_path=f'models/checkpoints/board_detector/{MODEL}/weight',
    #                                                       weights_save_folder=f'models/checkpoints/board_detector/{MODEL}',
    #                                                       json_file=json_file,
    #                                                       root_folder=root_folder,
    #                                                       from_pretrained=PRETRAINED,
    #                                                       writer=writer,
    #                                                       step=step,
    #                                                       device=device)
    #        # writer.add_hparams({'batch size': b, 'learning rate': l}, {'last_loss': last_loss, 'lowest_loss': lowest_loss})
    # writer.close()
    # print(f'last:{last_loss} lowest:{lowest_loss}')
    # print('Training Complete!')
    """
    BOARD DETECTOR VIEW OUTPUT
    """
    # weights = f'models/checkpoints/board_detector/{MODEL}/weight'
    #
    # board_detector = BoardDetector(model=MODEL).to(device)
    # board_detector.load_state_dict(torch.load(weights, map_location=device))
    # board_detector.eval()
    # board_data = BoardDetectorDataset(root_folder, json_file, size=320, target="points")
    #
    # tr = transforms.Compose([transforms.Resize(img_size)])
    # img_path = 'dataloader/data/raw/IMG_0624.jpg'
    # img = tr(read_image(img_path) / 255.0)
    # img = img.unsqueeze(0).to(device)
    # points = board_detector(img) * img_size
    # warped_img, _ = warp(img, reshape_coords(points), device=device)
    # fig, ax = plt.subplots(1, 2)
    # points = points[0].view(4, 2).transpose(0, 1)
    # ax[0].imshow(img[0].cpu().permute(1, 2, 0))
    # ax[0].scatter(points[0].detach().cpu(), points[1].detach().cpu())
    # ax[1].imshow(warped_img[0].detach().cpu().permute(1, 2, 0))
    # plt.show()

    # for i in range(len(board_data)):
    #     img, _ = board_data[i]
    #     img = img.unsqueeze(0).to(device)
    #     points = board_detector(img) * img_size
    #     warped_img, _ = warp(img, reshape_coords(points), device=device)
    #     fig, ax = plt.subplots(1, 2)
    #     points = points[0].view(4, 2).transpose(0, 1)
    #     ax[0].imshow(img[0].cpu().permute(1, 2, 0))
    #     ax[0].scatter(points[0].detach().cpu(), points[1].detach().cpu())
    #     ax[1].imshow(warped_img[0].detach().cpu().permute(1, 2, 0))
    #     plt.show()


if __name__ == '__main__':
    main()
