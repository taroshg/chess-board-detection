# installed imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm
import torch

# default imports
import math

# local imports
from detectors import PieceDetector
from dataloader import PieceDetectorDataset
from helpers import piece_detector_results


def train_piece_detector(weights_load_path=None, weights_save_folder=None, weights_name="weight",
                         json_file='dataloader/data/piece_data/piece_detection_coco.json',
                         img_size=(1000, 1000),
                         batch_size=2,
                         learning_rate=0.01,
                         weight_decay=1e-4,
                         epochs=10,
                         from_pretrained=True,
                         mixed_precision_training=False,
                         writer=None,
                         step=0,
                         device='cpu'):
    """
    This function trains the Piece Detector
    Args: 
        (str) weights_load_path: path to the Board Detector model weights that needs to be loaded
        (str) weights_save_folder: folder where the Board Detector weights will be saved
        (str) weights_name: name for the weights that will be saved
        (str) json_file: the location of the coco json file for detections
        (tuple) img_size: the size of input image
        (int) batch_size: size of each batch trained at once
        (float) learning_rate: lr of the model
        (int) epochs: # of iterations through dataset
        (bool) from_pretrained: should the Board Detector be loader with pretrained weights
        (bool) mixed_precision_training: reduces the float32 to float16, checkout pytorch documentation for more details
        (Summary Writer) writer: optional tensorboard summary writer, default location f'{weights_save_folder}/tensorboard'
        (int) step: the initial step to count from for writer (used for adding to tensorboard)

        in function args:
        (str) loss_save_path: path where losses plot fig (losses.jpg) will be stored

    Returns:
        loss, and the lowest_loss
    """

    piece_detector = PieceDetector(pretrained=from_pretrained).to(device)

    weights_save_path = weights_save_folder + f"/{weights_name}"
    loss_save_path = weights_save_folder + "/losses.jpg"

    if weights_save_folder is not None:
        weights_save_path = weights_save_folder + f"/{weights_name}"

    if from_pretrained:
        assert weights_load_path is not None
        piece_detector.load_state_dict(torch.load(weights_load_path, map_location=device))
        print('loaded weights for piece detector!')
    else:
        print('no weights are loaded for piece detector')

    if weights_save_path is not None:
        print('weights will be saved at' + weights_save_path + ' for piece detector')
    else:
        print('no weights will be saved for piece detector')

    # tensorboard writer initialization
    writer = SummaryWriter(f'{weights_save_folder}/tensorboard') if writer is None else writer

    optim = torch.optim.SGD(piece_detector.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
                            weight_decay=weight_decay)
    dataset = PieceDetectorDataset(json_file=json_file, size=img_size)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    # the collate_fn function is needed to create batches without using default stack mehtod,
    # the default collate_fn requires all boxes to be of same size, this is not possible since
    # each image has variable number of objects
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

    # mean average precision metric
    m_ap_metric = MeanAveragePrecision()

    lowest_loss = math.inf
    loss = math.inf
    step = step
    for epoch in tqdm(range(epochs)):
        for idx, (imgs, targets) in enumerate(dataloader):

            # train one step and get losses
            loss_dict, loss = train_one_step(imgs, targets, piece_detector, optim, scaler, device)

            # tensorboard related values
            writer.add_scalars("Loss", loss_dict, global_step=step)
            writer.add_scalar("Overall Loss", loss, global_step=step)
            step += 1

            if weights_save_folder is not None and loss.item() < lowest_loss:
                lowest_loss = loss
                torch.save(piece_detector.state_dict(), weights_save_path)

            # on last idx calculate mAP
            if idx == len(dataloader) - 1:
                img, target = dataset[0]
                targets = [{k: v.to(device) for k, v in target.items()}]
                piece_detector.eval()
                preds = piece_detector(img.unsqueeze(0).to(device))
                piece_detector.train()
                m_ap_metric.update(preds, targets)
                m_ap = m_ap_metric.compute()
        piece_detector.eval()
        real_out, out = piece_detector_results(4, piece_detector, dataset, device)
        writer.add_image('Prediction Outputs', make_grid([real_out, out]), step)
        piece_detector.train()

    return loss, lowest_loss, m_ap


def train_one_step(imgs, targets, piece_detector, optim, scaler, device):
    # converts tensor batch of images to an iterable list of images
    imgs = list(image.to(device) for image in imgs)

    # converts to a list of objects {boxes, labels, areas}
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # forward
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = piece_detector(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())

    # backward
    optim.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()

    return loss_dict, loss
