import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

# for use on the mask subset of celeba
class FaceDataset(Dataset):
    def __init__(self):
        self.imgs_path = "CelebAMask-HQ/CelebA-HQ-img"
        # i created this file myself that maps index to id for the 30000 in this set
        identities_path = "CelebAMask-HQ/CelebAMask-HQ-identities.txt"
        self.identities = pd.read_csv(identities_path)
        mapping = self.identities['identity'].value_counts().index
        # another way to map it might just be subtract 1 from all
        self.identities['id_mapped'] = 0
        for i, x in enumerate(self.identities['identity']):
            self.identities['id_mapped'][i] = mapping.get_loc(x)
        print('Finished initializing dataset')

    def __len__(self):
        return len(self.identities)

    def __getitem__(self, item):
        img_name = os.path.join(self.imgs_path, str(item) + ".jpg")
        image = Image.open(img_name)

        # image = np.array(image)
        # input_tensor = torch.tensor(image)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)

        return input_tensor, torch.tensor(self.identities['id_mapped'][item])

# for use on the full celeba
class FaceDatasetFull(Dataset):
    def __init__(self):
        self.imgs_basepath = "CelebA/CelebA"
        full_paths = glob.glob("CelebA/CelebA/*/*.jpg")
        self.df = pd.DataFrame(full_paths, columns=['img_path'])
        self.df['id'] = 0
        self.df['id_mapped'] = 0
        for index, x in enumerate(full_paths):
            id = os.path.dirname(x).split("/")[-1].split("-")[-1]
            # img_id = int(os.path.split(x)[-1].split('.')[0])
            self.df['id'][index] = id
        mapping = self.df['id'].value_counts().index
        # another way to map it might just be subtract 1 from all
        for i, x in enumerate(self.df['id']):
            self.df['id_mapped'][i] = mapping.get_loc(x)
        print('Finished initializing dataset')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_path = self.df['img_path'][item]
        image = Image.open(img_path)

        # image = np.array(image)
        # input_tensor = torch.tensor(image)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)


        return input_tensor, torch.tensor(self.df['id_mapped'][item])

class FaceDatasetFull2(Dataset):
    def __init__(self):
        self.imgs_basepath = "CelebA/CelebA"
        full_paths = glob.glob("CelebA/CelebA/*/*.jpg")
        self.df = pd.DataFrame(full_paths, columns=['img_path'])
        self.df['id'] = 0
        self.df['id_mapped'] = 0
        for index, x in enumerate(full_paths):
            id = os.path.dirname(x).split("/")[-1].split("-")[-1]
            # img_id = int(os.path.split(x)[-1].split('.')[0])
            self.df['id'][index] = id
        mapping = self.df['id'].value_counts().index
        # another way to map it might just be subtract 1 from all
        for i, x in enumerate(self.df['id']):
            self.df['id_mapped'][i] = mapping.get_loc(x)
        print('Finished initializing dataset')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, items):
        item = items[0]
        item2 = items[1]
        img_path = self.df['img_path'][item]
        image = Image.open(img_path)
        img_path2 = self.df['img_path'][item2]
        image2 = Image.open(img_path2)
        # image = np.array(image)
        # input_tensor = torch.tensor(image)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_tensor2 = preprocess(image2)

        same = 1. if self.df['id_mapped'][item] == self.df['id_mapped'][item2] else 0.

        return input_tensor, input_tensor2, same