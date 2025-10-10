import math
import torch
import numpy as np
from model.PHATNet import PHATNet
import os
import time
from torch.nn import functional as F
import torchvision
import cv2
import glob
import shutil

haze_path = '../dataset/PHATNet_NHHaze20_to_DenseHaze/haze'
clean_path = '../dataset/NHHaze20/train/gt'
output_path = '../dataset/PHATNet_NHHaze20_to_DenseHaze'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'train')
if not os.path.isdir(output_path):
    os.mkdir(output_path)
output_path = os.path.join(output_path, 'haze')
if not os.path.isdir(output_path):
    os.mkdir(output_path)

model_name = 'PHATNet_NHHaze20.pth'

net = PHATNet().cuda()
net.load_state_dict(torch.load(model_name))
net.cuda()

haze_name_path = sorted(glob.glob(os.path.join(haze_path, "*")))
clean_name_path = sorted(glob.glob(os.path.join(clean_path, "*")))


for j in range(len(clean_name_path)):
    clean = cv2.imread(clean_name_path[j]).astype(np.float32) / 255
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    clean = torch.from_numpy(clean.transpose((2, 0, 1))).unsqueeze(0).cuda()
    factor = 64
    h, w = clean.shape[2], clean.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    clean = F.pad(clean, (0, padw, 0, padh), 'reflect')
    H, W = clean.shape[2], clean.shape[3]
    video_name = (clean_name_path[j].split('/')[-1].split('_')[0]).zfill(2)
    video_path = os.path.join(output_path, video_name)
    if not os.path.isdir(video_path):
        os.mkdir(video_path)
    for idx, i in enumerate(range(len(haze_name_path))):
        name = haze_name_path[i].split('/')[-1].split('.')[0] + '.png'
        save_path = os.path.join(video_path, name)

        haze = cv2.imread(haze_name_path[i]).astype(np.float32) / 255
        haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
        haze = torch.from_numpy(haze.transpose((2, 0, 1))).unsqueeze(0).cuda()
        haze = F.pad(haze, (0, padw, 0, padh), 'reflect')
        with torch.no_grad():
            rehaze, _, _ = net(haze, clean)
        torchvision.utils.save_image(rehaze[:, :, :h, :w], save_path)
        print(save_path)