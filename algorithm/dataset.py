import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from parameter import label_mean, label_std

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None, label_transform=False):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.label_transform = label_transform
        self.fixed_image_path = "/home/zty/data/arr_code_0326/origin/003.jpg"

    def __len__(self):
        return len(self.images_path)
    
    # def label_Normalize(self, list, mean, std):
    #     list =np.array(list)
    #     mean=np.array(mean)
    #     std=np.array(std)
    #     list = (list - mean) / std
    #     list = list.tolist()
    #     return list


    def label_Normalize(self,data: list, mean: list, std: list) -> list:

        if len(data) != len(mean) or len(data) != len(std):
            raise ValueError("same length")

        std = np.array(std)
        zero_std_indices = np.where(std == 0)[0]  

        data = np.array(data)
        mean = np.array(mean)

        non_zero_std_indices = np.where(std != 0)[0]
        normalized_data = data.copy()
        normalized_data[non_zero_std_indices] = (data[non_zero_std_indices] - mean[non_zero_std_indices]) / std[non_zero_std_indices]

        return normalized_data.tolist()

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        img_orin = Image.open(self.fixed_image_path).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
            img_orin = self.transform(img_orin)

        if self.label_transform is not False:
            label = self.label_Normalize(label, label_mean, label_std)

        img = torch.cat((img, img_orin), 0)

        return img, label


