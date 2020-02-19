import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from pathlib import Path
from PIL import Image


class ImageListDataset(Dataset):
    """"""

    def __init__(self, imgs, transform=None):
        self.root_dir = None
        self.files = imgs
        self.transform = transform

    def __getitem__(self, index):
        im = Image.open(self.files[index])
        if self.transform:
            im = self.transform(im)

        return im, 0, index

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__
        body = ["Number of Images: {}".format(self.__len__())]
        if self.root_dir is not None:
            body.append("Root location: {}".format(self.root_dir))
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)


def get_dataset(image_pairs, root_dir=Path('images')):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    imgs = [root_dir/img for imgs in image_pairs for img in imgs]
    dataset = ImageListDataset(imgs=imgs, transform=transform)

    return dataset


def show_grid(dataset):
    imgs = []
    for image_num in range(0, len(dataset), 2):
        imgs.append(np.hstack(
            [np.array(dataset[image_num][0]), np.array(dataset[image_num+1][0])]))
    imgs = np.vstack(imgs)

    return Image.fromarray(imgs)
