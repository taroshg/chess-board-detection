import labelbox
import json
import requests

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

def download_data(json_file):
    """
        Downloads data from labelbox
    """
    # LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDBzb2N1anFhdW9sMHo4dzFkbzQyY2U5Iiwib3JnYW5pemF0aW9uSWQiOiJjbDBzb2N1amhhdW9rMHo4dzI4bmU5NzJxIiwiYXBpS2V5SWQiOiJjbDllamswaXgwOWptMDc0MjQyNDAxaDlvIiwic2VjcmV0IjoiOGNjNjJjYTlkNGQ0NDRkMWQxNmI4YzAxZDEwMDQ5MTMiLCJpYXQiOjE2NjYxMTc4NjAsImV4cCI6MjI5NzI2OTg2MH0.8v_rvvYPgpvdvdLujGCQr0vxqirdhYxbHHLxsVf8Ztw"
    # lb = labelbox.Client(api_key=LB_API_KEY)
    # # Get project by ID
    # project = lb.get_project('cl9eifuyd11va081ccoxt8nff')
    # # Export image and text data as an annotation generator:
    # labels = project.label_generator()
    # # Export all labels as a json file:
    # labels = project.export_labels(download = True)

    # with open(json_file, "w") as outfile:
    #     outfile.write(str(labels))


    DATA_LOCATION = 'data'

    f = open(json_file)
    data = json.load(f)

    for i, elm in enumerate(data):
        img_url = elm["Labeled Data"]
        img_data = requests.get(img_url).content
        with open(f'{DATA_LOCATION}/{i}.jpg', 'wb') as handle:
            handle.write(img_data)

    f.close()

# download_data("data_1-14-2023.json") # Download data!

# original size: 320 x 320
device = "cuda" if torch.cuda.is_available() else "cpu"
class Bd_Data(Dataset):
    def __init__(self, json_file="./data_1-14-2023.json", size=(320, 320)) -> None:
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Resize(size)
        super().__init__()
        print("Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        coords = self.data[i]["Label"]["objects"][0]["polygon"]
        img = self.tr(read_image(f'data/{i}.jpg') / 255.0).to(device) # applies transfromations to img
        h = self.s[0]
        w = self.s[1]
        # gets the coords and normalizes it to 0 - 1.
        label = torch.tensor([coords[0]['x'] / h, coords[0]['y'] / w, coords[1]['x'] / h, coords[1]['y'] / w,
                             coords[2]['x'] / h, coords[2]['y'] / w, coords[3]['x']/ h, coords[3]['y'] / w]).to(device);
        return img, label