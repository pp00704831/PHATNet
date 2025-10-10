import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random

class ToTensor(object):
    def __call__(self, data):

        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()

        return data
class Train_Loader(Dataset):
    def __init__(self, haze_path, clean_path):
        self.haze_name_list = []
        self.clean_name_list = []
        self.transform = transforms.Compose([ToTensor()])

        haze_name_path = sorted(glob.glob(os.path.join(haze_path, "*")))
        clean_name_path = sorted(glob.glob(os.path.join(clean_path, "*")))

        for i in range(len(haze_name_path)):
            self.haze_name_list.append(haze_name_path[i])
            self.clean_name_list.append(clean_name_path[i])

        assert len(self.haze_name_list) == len(self.clean_name_list), "Missmatched Length!"

    def select_idx(self, idx):
        while True:
            select_idx = random.randint(0, len(self.haze_name_list)-1)
            if select_idx != idx:
                return select_idx

    def __len__(self):
        return len(self.haze_name_list)

    def __getitem__(self, idx):
        haze = cv2.imread(self.haze_name_list[idx]).astype(np.float32) / 255
        haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
        clean = cv2.imread(self.clean_name_list[idx]).astype(np.float32) / 255
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        select_idx = self.select_idx(idx)
        clean_2 = cv2.imread(self.clean_name_list[select_idx]).astype(np.float32) / 255
        clean_2 = cv2.cvtColor(clean_2, cv2.COLOR_BGR2RGB)
        sample = {'haze': haze,
                  'clean': clean,
                  'clean_2': clean_2}

        if self.transform:
            sample = self.transform(sample)

        return sample