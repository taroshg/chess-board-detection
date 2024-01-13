import matplotlib.pyplot as plt

from dataloader import BoardDetectorDataset, PieceDetectorDataset
from models import PieceDetector, BoardDetector
from kornia.geometry.transform.imgwarp import get_perspective_transform
from helpers import warp, reshape_coords
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from kornia.geometry import transform_points
from torchvision import transforms
from PIL import Image
from torchvision.io import read_image

device = "cpu"
if torch.cuda.is_available():
    print('running cuda...')
    device = "cuda"
elif torch.backends.mps.is_available():
    print('running mps...')
    device = "mps"


def main():
    # img, bboxes, labels, coords = get_predictions('dataloader/data/board_data/IMG_0576_resized.jpg')
    img, bboxes, labels, coords = get_actual(4)

    # gets anchor point by adding the height and width to coordinate
    points = bboxes_to_points(bboxes)

    warped_img, M = warp(img, reshape_coords(coords), device=device)
    out = transform_points(M, points)

    calculateFEN(out[0], labels)

    # out = out.transpose(1, 2)[0]
    # plt.imshow(warped_img[0].to('cpu').to(torch.uint8).permute(1, 2, 0))
    # plt.scatter(out[0].detach().to('cpu'), out[1].detach().to('cpu'))
    # for i, label in enumerate(labels):
    #     plt.annotate(label, (out[0][i], out[1][i]), color='red')
    # for i in range(8):
    #     plt.axhline(y=i * 40, color='black', linewidth=1)
    #     plt.axvline(x=i * 40, color='black', linewidth=1)
    # plt.xlim([0, 320])
    # plt.ylim([320, 0])
    # plt.show()


def prep_image(image_path, size, device='cpu'):
    tr = transforms.Resize(size)
    img = tr(read_image(image_path) / 255.0)
    return img.unsqueeze(0).to(device)


def get_board_detector(weights=None, model='resnet', device='cpu'):
    board_detector = BoardDetector(model=model).to(device)
    if weights is not None:
        board_detector.load_state_dict(torch.load(weights, map_location=device))
    board_detector.eval()
    return board_detector


def get_piece_detector(weights=None, device='cpu'):
    piece_detector = PieceDetector().to(device)
    if weights is not None:
        piece_detector.load_state_dict(torch.load(weights, map_location=device))
    piece_detector.eval()
    return piece_detector


def get_predictions(img_path, threshold=0.5, img_size=320,
                    piece_weight='models/checkpoints/piece_detector/320_faster_rcnn/weight',
                    board_weight='models/checkpoints/board_detector/squeezenet/weight'):
    img = prep_image(img_path, size=(img_size, img_size), device=device)
    board_detector = get_board_detector(board_weight, device=device)
    piece_detector = get_piece_detector(piece_weight, device=device)

    pieces = ['none', 'P', 'N', 'B', 'R', 'Q', 'K',
              'p', 'n', 'b', 'r', 'q', 'k']

    coords = board_detector(img)[0]
    boxes, labels, scores = piece_detector(img)[0].values()

    img = img * 255
    coords = (coords * img_size).detach()

    boxes = boxes[scores > threshold, :]  # filter boxes by scores
    labels = labels[scores > threshold]  # filter labels by scores
    labels = [pieces[label] for label in labels.tolist()]  # ints to string labels

    return img, boxes, labels, coords


def get_actual(idx, img_size=320, piece_dataset=None, board_dataset=None):
    board_dataset = BoardDetectorDataset(root='dataloader/data/raw/',
                                         json_file='dataloader/data/board_detector_coco.json',
                                         size=img_size) if board_dataset is None else board_dataset
    piece_dataset = PieceDetectorDataset(root='dataloader/data/raw/',
                                         json_file='dataloader/data/piece_detection_coco.json',
                                         size=(img_size, img_size)) if piece_dataset is None else piece_dataset

    pieces = ['none', 'P', 'N', 'B', 'R', 'Q', 'K',
              'p', 'n', 'b', 'r', 'q', 'k']

    _, coords = board_dataset[idx]
    img, target = piece_dataset[idx]
    bboxes = target['boxes']
    labels = target['labels']

    img = (img * 255).unsqueeze(0).to(device)
    coords = (coords * img_size).detach().to(device)
    labels = [pieces[label] for label in labels.tolist()]
    return img, bboxes, labels, coords


def bboxes_to_points(bboxes : torch.Tensor, offset : float = 2):
    """

    Args:
        offset (float): percentage of height of all bounding boxes
        bboxes (torch.Tensor): bboxes of size (batch, n_boxes, 4)

    Returns:
        points (torch.Tensor): points of size (batch, n_boxes, 2)
    """
    bboxes = box_convert(bboxes, 'xyxy', 'xywh').unsqueeze(0).to(device)

    adjusted_bboxes = bboxes[:, :, 3:4] * offset
    adjusted_bboxes = torch.cat([bboxes[:, :, 0:3], adjusted_bboxes], dim=-1)
    points = adjusted_bboxes[:, :, :2] + (adjusted_bboxes[:, :, 2:4] // 2)
    return points


def calculateFEN(points, labels, img_size=320):
    chessboard = [[' ' for i in range(8)] for j in range(8)]
    cell_size = img_size // 8
    points = torch.clamp(points.to(int), 1, img_size - 1)  # clamping is done to not have points on the edge
    for i, point in enumerate(points):
        label = labels[i]
        grid_coord = point // cell_size
        x = grid_coord[0].item()
        y = grid_coord[1].item()
        if x < 8 and y < 8:
            chessboard[y][x] = label
    fen = fen_from_matrix(chessboard)
    return fen


def print_grid(grid):
    for row in grid:
        print(' '.join(row))


def fen_from_matrix(matrix):
    fen = ''
    for row in matrix:
        empty = 0
        for cell in row:
            if cell == ' ':
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += cell
        if empty > 0:
            fen += str(empty)
        fen += '/'
    return fen[:-1]


def get_url(fen):
    return f'https://lichess.org/editor/{fen}_w_-_-_0_1?color=white'


if __name__ == '__main__':
    main()


