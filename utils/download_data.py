# local imports
import requests
import json

def download_data(json_file, location):
    """
        Downloads data from labelbox server using downloaded json file
    """
    f = open(json_file)
    data = json.load(f)

    for i, elm in enumerate(data):
        img_url = elm["Labeled Data"]
        img_data = requests.get(img_url).content
        with open(f'{location}/{i}.jpg', 'wb') as handle:
            handle.write(img_data)

    f.close()

# download_data("data_1-14-2023.json", "data") # Download data!