import os
import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import glob

out_path = './output/focalnet_PHATNet_NHHaze20_to_OHaze'
gt_path = '../dataset/OHaze/gt'

totalo_psnr = 0
total_ssim = 0

def calc_psnr(result, gt):
    mse = np.mean(np.power((result - gt), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calc_ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32)
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32)
    img2 = img2.transpose((2, 0, 1))

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)



output_name_path = sorted(glob.glob(os.path.join(out_path, "*")))
gt_name_path = sorted(glob.glob(os.path.join(gt_path, "*")))


for i in range(len(output_name_path)):
    output = cv2.imread(output_name_path[i]).astype(np.float32) / 255
    gt = cv2.imread(gt_name_path[i]).astype(np.float32) / 255
    psnr = calc_psnr(output, gt)
    ssim = calc_ssim(output, gt)
    totalo_psnr += psnr
    total_ssim += ssim
    print(i+1, totalo_psnr / (i+1), total_ssim / (i+1))
