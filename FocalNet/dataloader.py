import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data

class ToTensor(object):
    def __call__(self, data):

        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()

        return data

class Train_Loader_Test_Time(Dataset):
    def __init__(self, haze_path, clean_path, crop_size):
        self.haze_name_list = []
        self.clean_name_list = []
        self.transform = transforms.Compose([RandomCrop(crop_size[0], crop_size[1]), ToTensor()])

        for video in os.listdir(haze_path):
            haze_name_path = sorted(glob.glob(os.path.join(haze_path, video, "*")))
            gt_path = os.path.join(clean_path, video + '_GT.png')
            for i in range(len(haze_name_path)):
                self.haze_name_list.append(haze_name_path[i])
                self.clean_name_list.append(gt_path)

        assert len(self.haze_name_list) == len(self.clean_name_list), "Missmatched Length!"

    def __len__(self):
        return len(self.haze_name_list)

    def __getitem__(self, idx):

        haze = cv2.imread(self.haze_name_list[idx]).astype(np.float32) / 255
        haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
        clean = cv2.imread(self.clean_name_list[idx]).astype(np.float32) / 255
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        sample = {'haze': haze,
                  'clean': clean}

        if self.transform:
            sample = self.transform(sample)

        return sample