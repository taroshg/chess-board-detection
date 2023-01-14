import torch
import torch.nn as nn
from model import Bd_Model # Bd is short for board detection
from data import Bd_Data # Bd (Board Detection)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt

def main():
    CURRENT_TRAINING_VERSION = 5
    SAVE_PATH = f'./checkpoint/{CURRENT_TRAINING_VERSION}/model'
    LOAD_PATH = f'./checkpoint/{CURRENT_TRAINING_VERSION - 1}/model'
    LOAD_MODEL = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bd_model = Bd_Model().to(device)
    if LOAD_MODEL : 
        bd_model.load_state_dict(torch.load(LOAD_PATH)) 
        print('loaded model!')

    # https://twitter.com/karpathy/status/801621764144971776?lang=en (The Andrej Karpathy constant = 3e-4)
    LEARNING_RATE = 3e-4
    optim = torch.optim.Adam(bd_model.parameters(), lr=LEARNING_RATE) 
    dataset = Bd_Data()

    BATCH_SIZE = 64
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    EPOCHS = 30
    criterion = nn.MSELoss()

    losses = []
    lowest_loss = math.inf
    for epoch in tqdm(range(EPOCHS)):
        for data in dataloader:
            x, y = data

            optim.zero_grad()

            out = bd_model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()

            losses += [loss.item()]
            if(loss.item() < lowest_loss):
                lowest_loss = loss.item()
                torch.save(bd_model.state_dict(), SAVE_PATH)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
