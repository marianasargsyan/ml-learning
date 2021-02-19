import numpy as np
import torch
from torch.utils.data import Dataset
from imageio import imread
import os
import shutil


def prepare_dataset(directory1, directory2, parent_dir):
    path1 = os.path.join(parent_dir, directory1)
    path2 = os.path.join(parent_dir, directory2)
    if not os.path.exists(path1):
        os.mkdir(path1)
    elif not os.path.exists(path2):
        os.mkdir(path2)
    print(f'Directories named {directory1} and {directory2} created in {parent_dir}')
    file_list = os.listdir(parent_dir)
    for image in file_list:
        if directory1.lower() in image.lower():
            shutil.move(image, path1)
        elif directory2.lower() in image.lower():
            shutil.move(image, path2)
    print(f'Files moved to corresponding folders.')


def read_dataset(path, folder_size, img_height, img_width):
    images_path = f"{path}/images"
    labels_path = f"{path}/labels"

    images = np.zeros((folder_size, img_height, img_width))
    labels1 = np.zeros((folder_size, img_height, img_width))
    labels2 = np.zeros((folder_size, img_height, img_width))

    for i in range(folder_size):
        img_file_path = f"{images_path}/image_{i}.png"
        lbl_file_path1 = f"{labels_path}/dog/0{i}.png"
        lbl_file_path2 = f"{labels_path}/cat/1{i}.png"

        images[i] = imread(img_file_path)
        labels1[i] = imread(lbl_file_path1)
        labels2[i] = imread(lbl_file_path2)

    return images, labels1, labels2


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    prepare_dataset('Cat', 'Dog', 'Data/dataset')
