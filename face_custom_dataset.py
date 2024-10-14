import os
import torch
import pandas as pd
import numpy as np
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2




class FaceLandmarksDataset(Dataset):
    def __init__(self, root_dir ,image_size = 128, transform=None,):
        self.root_dir: Path = Path(root_dir)
        self.transform = transform

        self.image_size=image_size
        data = self.set_data_paths(root_dir, ["*.jpg", "*.png"])
        self.df = self.make_csv(data)
        self.num_of_classes = self.df['id'].nunique()
        print(f"num of classes: {self.num_of_classes}")
        self.df[["name", "id"]].groupby(["name", "id"]).mean().reset_index().to_csv("./name_id.csv")

    def set_data_paths(self, root_dir, extensions: list = list) -> dict:
        all_names = os.listdir(root_dir)
        data_paths = {}
        for name in all_names:
            new_path = self.root_dir.joinpath(name)
            paths = []
            for ext in extensions:
                paths += list(new_path.rglob(ext))
            data_paths[name] = paths
        return data_paths

    def make_csv(self, data: dict):
        records = []
        id_count = {}
        for name, paths in data.items():
            for path in paths:
                if path.exists():
                    new_path = str(path)
                    id = int(new_path.split("\\")[1].split("_")[0])
                    if id not in id_count:
                        id_count[id] = id
                    records.append({
                        "image_path": new_path,
                        "name": name,
                        "id": id
                    })

        df = pd.DataFrame(records)
        print(id_count)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]["image_path"]
        id = self.df.iloc[idx]["id"]
        if id < 0 or id > self.num_of_classes:
            raise ValueError(f"Invalid label {id} for the number of classes {self.num_of_classes}.")
        # Open the image
        image = Image.open(img_path)


        # Check if the image is grayscale, if so, convert it to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        #image.save(f"./images/{idx}.jpg")

        # Resize the image to ensure it's exactly 128x128 using LANCZOS filter
        #image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert the image to a NumPy array
        image = np.array(image)


        # Ensure that the image has been properly read
        assert image.size != 0, f"[ERROR] The image {img_path} is empty or cannot be read."

        #image = np.transpose(image, (2,0,1))



        if self.transform:
            image = self.transform(image)
        # Create a sample dictionary
        sample = {'id': id, 'image': image}


        # image_np = image.cpu().numpy().transpose(1, 2, 0)
        #
        # # Ensure the values are in the correct range
        # image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        #
        # # Save the image using plt.imsave()
        # plt.imsave(f"./images/{idx}.jpg", image_np)


        return sample

