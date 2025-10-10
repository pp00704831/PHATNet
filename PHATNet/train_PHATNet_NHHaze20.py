import math
import torch
from dataloader import Train_Loader
import random
from torch.utils.data import DataLoader
import numpy as np
from model.PHATNet import PHATNet
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

cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="NCCL", init_method='env://')

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
haze_path = '../dataset/NHHaze20/train/haze'
clean_path = '../dataset/NHHaze20/train/gt'
model_name = 'PHATNet_NHHaze20'
start_epoch = 0
end_epoch = 1000
check_point_epoch = 500
batch_size = 1
init_lr = 1e-4
min_lr = 1e-7
net = PHATNet()

writer = SummaryWriter(model_name)

if os.path.exists('last_{}.pth'.format(model_name)):
    print('load_pretrained')
    training_state = torch.load('last_{}.pth'.format(model_name), map_location=torch.device("cpu"))
    start_epoch = training_state['epoch'] + 1
    new_weight = net.state_dict()
    new_weight.update(training_state['model_state'])
    net.load_state_dict(new_weight)

net.to(device)
num_gpus = torch.cuda.device_count()

net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank, find_unused_parameters=True)
# Traning loader
Train_set = Train_Loader(haze_path, clean_path)
train_sampler = DistributedSampler(Train_set)
dataloader_train = DataLoader(Train_set, sampler=train_sampler, batch_size=batch_size // num_gpus,
                              num_workers=1, pin_memory=True)

# Model and optimizer
optimizer = optim.Adam(net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=end_epoch, eta_min=min_lr)

if os.path.exists('last_{}.pth'.format(model_name)):
    new_optimizer = optimizer.state_dict()
    new_optimizer.update(training_state['optimizer_state'])
    optimizer.load_state_dict(new_optimizer)
    new_scheduler = scheduler.state_dict()
    new_scheduler.update(training_state['scheduler_state'])
    scheduler.load_state_dict(new_scheduler)


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


for epoch in range(start_epoch, end_epoch):
    train_sampler.set_epoch(epoch)

    tq = tqdm.tqdm(dataloader_train, total=len(dataloader_train))
    tq.set_description(
        'Epoch {}, lr {:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))

    total_train_loss = 0.
    total_train_rehaze_loss = 0.
    total_train_reclean_loss = 0.
    total_train_psnr = 0.

    for idx, sample in enumerate(tq):
        # input: [B, 4, C, H, W], gt: [B, 17, C, H, W]

        haze, clean, clean_2 = sample['haze'].to(device), sample['clean'].to(device), sample['clean_2'].to(device)

        factor = 64
        h, w = haze.shape[2], haze.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        haze = F.pad(haze, (0, padw, 0, padh), 'reflect')
        clean_2 = F.pad(clean_2, (0, padw, 0, padh), 'reflect')
        clean = F.pad(clean, (0, padw, 0, padh), 'reflect')
        H, W = haze.shape[2], haze.shape[3]

        haze_down_1 = F.interpolate(haze, scale_factor=0.5, mode='bilinear')
        haze_down_2 = F.interpolate(haze, scale_factor=0.25, mode='bilinear')
        clean_down_1 = F.interpolate(clean, scale_factor=0.5, mode='bilinear')
        clean_down_2 = F.interpolate(clean, scale_factor=0.25, mode='bilinear')
        rehaze_0, rehaze_1, rehaze_2 = net(haze, clean)
        reclean_0, reclean_1, reclean_2 = net(clean_2, clean)

        optimizer.zero_grad()

        loss_rehaze_0 = nn.L1Loss()(rehaze_0, haze)
        loss_rehaze_1 = nn.L1Loss()(rehaze_1, haze_down_1)
        loss_rehaze_2 = nn.L1Loss()(rehaze_2, haze_down_2)
        loss_reclean_0 = nn.L1Loss()(reclean_0, clean)
        loss_reclean_1 = nn.L1Loss()(reclean_1, clean_down_1)
        loss_reclean_2 = nn.L1Loss()(reclean_2, clean_down_2)

        loss_rehaze = loss_rehaze_0 + loss_rehaze_1 + loss_rehaze_2
        loss_reclean = loss_reclean_0 + loss_reclean_1 + loss_reclean_2

        loss = loss_rehaze + loss_reclean
        loss.backward()
        optimizer.step()

        psnr = calc_psnr(rehaze_0.detach(), haze.detach())
        total_train_loss += loss.item()
        total_train_rehaze_loss += loss_rehaze.item()
        total_train_reclean_loss += loss_reclean.item()
        total_train_psnr += psnr

        tq.set_postfix(P=(total_train_psnr / (idx + 1)),
                       L=(total_train_loss / (idx + 1)),
                       L_H=(total_train_rehaze_loss / (idx + 1)),
                       L_C=(total_train_reclean_loss / (idx + 1)))

        # print('Reconstruction:', total_train_reconstruction_loss / (idx + 1))
        # print('style_loss:', total_train_style_loss / (idx + 1))

    scheduler.step()
    writer.add_scalar('Train_loss', total_train_loss / (idx + 1), epoch)
    writer.add_scalar('Train_psnr', total_train_psnr / (idx + 1), epoch)
    writer.add_scalar('Train_rehaze', total_train_rehaze_loss / (idx + 1), epoch)
    writer.add_scalar('Train_reclean', total_train_reclean_loss / (idx + 1), epoch)

    if args.local_rank == 0:
        # save parameters
        scheduler_state = scheduler.state_dict()
        optimizer_state = optimizer.state_dict()
        net_state = net.module.state_dict()
        training_state = {'epoch': epoch, 'model_state': net_state,
                          'scheduler_state': scheduler_state, 'optimizer_state': optimizer_state}
        torch.save(training_state, 'last_{}.pth'.format(model_name))

        if (epoch % check_point_epoch) == 0:
            torch.save(training_state, 'epoch_{}_{}.pth'.format(epoch, model_name))

        if epoch == (end_epoch - 1):
            torch.save(net_state, 'final_{}.pth'.format(model_name))
