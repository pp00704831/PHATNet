import math
import torch
from dataloader import Train_Loader_Test_Time
import random
from torch.utils.data import DataLoader
import numpy as np
from model.FocalNet import FocalNet
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2
from tensorboardX import SummaryWriter
import os
from torch.utils.data.distributed import DistributedSampler
import argparse
import time
import torch.nn.functional as F
import glob
cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="gloo", init_method='env://')

# training seed
seed = 777 + args.local_rank
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# hyperparameters
haze_path = '../dataset/PHATNet_NHHaze20_to_IHaze/train/haze'
clean_path = '../dataset/NHHaze20/train/gt'
model_name = 'focalnet_PHATNet_NHHaze20_to_IHaze'
crop_size = [800, 1200]
start_epoch = 0
end_epoch = 1
check_point_epoch = 10
batch_size = 5
init_lr = 1e-6
min_lr = 1e-7
net = FocalNet()

if args.local_rank == 0:
    writer = SummaryWriter(model_name)

new_weight = net.state_dict()
new_weight.update(torch.load('focalnet_baseline.pth', map_location=torch.device("cpu")))
net.load_state_dict(new_weight)

net.to(device)
num_gpus = torch.cuda.device_count()

net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
# Traning loader
Train_set = Train_Loader_Test_Time(haze_path, clean_path, crop_size)
train_sampler = DistributedSampler(Train_set)
dataloader_train = DataLoader(Train_set, sampler=train_sampler, batch_size=batch_size // num_gpus,
                              num_workers=1, pin_memory=True)

# Model and optimizer
optimizer = optim.Adam(net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=end_epoch, eta_min=min_lr)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

def calc_psnr(result, gt):
    result = result.cpu().numpy()
    gt = gt.cpu().numpy()
    mse = np.mean(np.power((result - gt), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def FFT_Loss(x, y):

    x_fft = torch.fft.fft2(x, dim=(-2, -1))
    x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
    y_fft = torch.fft.fft2(y, dim=(-2, -1))
    y_fft = torch.stack((y_fft.real, y_fft.imag), -1)
    fft_loss = nn.L1Loss()(x_fft, y_fft)

    return fft_loss

for epoch in range(start_epoch, end_epoch):
    train_sampler.set_epoch(epoch)
    tq = tqdm.tqdm(dataloader_train, total=len(dataloader_train))
    tq.set_description(
        'Epoch {}, lr {:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))

    total_train_loss = 0.
    total_train_psnr = 0.
    total_test_psnr = 0
    for idx, sample in enumerate(tq):
        # input: [B, 4, C, H, W], gt: [B, 17, C, H, W]

        haze, clean = sample['haze'].to(device), sample['clean'].to(device)
        clean2 = F.interpolate(clean, scale_factor=0.5, mode='bilinear')
        clean4 = F.interpolate(clean, scale_factor=0.25, mode='bilinear')
        output = net(haze)

        optimizer.zero_grad()
        loss_0 = nn.L1Loss()(output[0].clamp(0, 1), clean4)
        loss_1 = nn.L1Loss()(output[1].clamp(0, 1), clean2)
        loss_2 = nn.L1Loss()(output[2].clamp(0, 1), clean)
        loss_fft_0 = FFT_Loss(output[0], clean4)
        loss_fft_1 = FFT_Loss(output[1], clean2)
        loss_fft_2 = FFT_Loss(output[2], clean)
        loss_fft = loss_fft_0 + loss_fft_1 + loss_fft_2
        loss_reconstruct = loss_0 + loss_1 + loss_2
        loss = (loss_reconstruct + 0.1 * loss_fft)
        loss.backward()
        optimizer.step()


        psnr = calc_psnr(output[2].detach(), clean.detach())
        total_train_loss += loss.item()
        total_train_psnr += psnr

        tq.set_postfix(PSNR=(total_train_psnr / (idx + 1)),
                       Loss=(total_train_loss / (idx + 1)))

        # print('Reconstruction:', total_train_reconstruction_loss / (idx + 1))
        # print('style_loss:', total_train_style_loss / (idx + 1))

    scheduler.step()

    scheduler_state = scheduler.state_dict()
    optimizer_state = optimizer.state_dict()
    net_state = net.module.state_dict()

    if args.local_rank == 0:
        writer.add_scalar('Train_loss', total_train_loss / (idx + 1), epoch)
        writer.add_scalar('Train_PSNR', total_train_psnr / (idx + 1), epoch)

    if args.local_rank == 0:
        scheduler_state = scheduler.state_dict()
        optimizer_state = optimizer.state_dict()
        net_state = net.module.state_dict()

        # save parameters
        training_state = {'epoch': epoch, 'model_state': net_state,
                          'scheduler_state': scheduler_state, 'optimizer_state': optimizer_state}
        torch.save(training_state, 'last_{}.pth'.format(model_name))

        if (epoch % check_point_epoch) == 0:
            torch.save(training_state, 'epoch_{}_{}.pth'.format(epoch, model_name))

        if epoch == (end_epoch - 1):
            torch.save(net_state, 'final_{}.pth'.format(model_name))

