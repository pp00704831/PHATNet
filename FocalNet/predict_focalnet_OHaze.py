import math
import torch
import numpy as np
from model.FocalNet import FocalNet
import os
import time
from torch.nn import functional as F
import torchvision
import cv2
import glob


haze_path = '../dataset/OHaze/haze'
clean_path = '../dataset/OHaze/gt'
if not os.path.exists('output'):
    os.makedirs('output')
out_path = './output/focalnet_PHATNet_NHHaze20_to_OHaze'
model_name = './focalnet_PHATNet_NHHaze20_to_OHaze.pth'

if not os.path.isdir(out_path):
        os.mkdir(out_path)

net = FocalNet().cuda()
net.load_state_dict(torch.load(model_name))
net.cuda()

haze_name_path = sorted(glob.glob(os.path.join(haze_path, "*")))
clean_name_path = sorted(glob.glob(os.path.join(clean_path, "*")))

for i in range(len(haze_name_path)):
    haze = cv2.imread(haze_name_path[i]).astype(np.float32) / 255
    haze = cv2.cvtColor(haze, cv2.COLOR_BGR2RGB)
    haze = torch.from_numpy(haze.transpose((2, 0, 1))).unsqueeze(0).cuda()

    clean = cv2.imread(clean_name_path[i]).astype(np.float32) / 255
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    clean = torch.from_numpy(clean.transpose((2, 0, 1))).unsqueeze(0).cuda()

    name = haze_name_path[i].split('/')[-1]
    save_path = os.path.join(out_path, name)
    with torch.no_grad():
        output = net(haze)
        output = output[2].clamp(0, 1)
        torchvision.utils.save_image(output, save_path)

print(model_name)