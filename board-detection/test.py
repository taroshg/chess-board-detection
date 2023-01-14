from asyncore import read
import torch
import torch.nn as nn
from model import Bd_Model # Bd is short for board detection
from data import Bd_Data # Bd (Board Detection)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

SAVE = False # Toggle this to avoid override

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Bd_Data()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Bd_Model().to(device)
    VERSION = 4
    LOAD_PATH = f'./checkpoint/{VERSION}/model'
    model.load_state_dict(torch.load(LOAD_PATH))
    model.eval()
    
    for x, y in dataloader:
        out = model(x)

        pred_coords = (out[0] * 320).tolist()
        real_coords = (y[0] * 320).tolist()

        print(pred_coords)
        print(real_coords)
        
        plt.imshow(x[0].cpu().permute(1, 2, 0))
        plt.plot(pred_coords[::2]+[pred_coords[0]], pred_coords[1::2]+[pred_coords[1]], '.r-')
        plt.plot(real_coords[::2]+[real_coords[0]], real_coords[1::2]+[real_coords[1]], '.g-', alpha=0.5)
        if SAVE:
            plt.savefig(f'./checkpoint/{VERSION}/out.png')
        plt.show()
        quit()



if __name__ == '__main__':
    main()