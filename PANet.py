import torch
import torch.nn as nn
import logging
import sys
from torch.nn import functional as F
import math
from thop import profile


class Encoder_A(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Encoder_A, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 6

        self.en_1_input = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation))

        self.en_2_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_1, dim_2, kernel_size=3, padding=1),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation))

        self.en_3_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_2, dim_3, kernel_size=3, padding=1),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation))

        self.en_4_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_3, dim_4, kernel_size=3, padding=1),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            nn.ReLU())

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):

        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)

        hx = torch.exp(-self.pool(self.output(hx)))

        return hx

class Encoder_T(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Encoder_T, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 6

        self.en_1_input = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation))

        self.en_2_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_1, dim_2, kernel_size=3, padding=1),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation))

        self.en_3_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_2, dim_3, kernel_size=3, padding=1),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation))

        self.en_4_input = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(dim_3, dim_4, kernel_size=3, padding=1),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            nn.ReLU())

    def forward(self, x):

        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)

        hx = torch.exp(-self.output(hx))

        return hx


class Encoder_J(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Encoder_J, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_en_1 = 2
        self.number_en_2 = 2
        self.number_en_3 = 2
        self.number_en_4 = 6

        self.en_1_input = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_1_res = nn.ModuleList()
        for i in range(self.number_en_1):
            self.en_1_res.append(nn.Sequential(
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
                self.activation))

        self.en_2_input = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_2_res = nn.ModuleList()
        for i in range(self.number_en_2):
            self.en_2_res.append(nn.Sequential(
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
                self.activation))

        self.en_3_input = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_3_res = nn.ModuleList()
        for i in range(self.number_en_3):
            self.en_3_res.append(nn.Sequential(
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
                self.activation))

        self.en_4_input = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_4_res = nn.ModuleList()
        for i in range(self.number_en_4):
            self.en_4_res.append(nn.Sequential(
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                self.activation,
                nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
                self.activation))


    def forward(self, x):

        hx = self.en_1_input(x)
        for i in range(self.number_en_1):
            hx = self.activation(self.en_1_res[i](hx) + hx)
        res_1 = hx

        hx = self.en_2_input(hx)
        for i in range(self.number_en_2):
            hx = self.activation(self.en_2_res[i](hx) + hx)
        res_2 = hx

        hx = self.en_3_input(hx)
        for i in range(self.number_en_3):
            hx = self.activation(self.en_3_res[i](hx) + hx)
        res_3 = hx

        hx = self.en_4_input(hx)
        for i in range(self.number_en_4):
            hx = self.activation(self.en_4_res[i](hx) + hx)

        return hx, res_1, res_2, res_3

class Decoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4):
        super(Decoder, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.number_de_1 = 2
        self.number_de_2 = 2
        self.number_de_3 = 2
        self.number_de_4 = 6

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.de_4_res = nn.ModuleList()
        for i in range(self.number_de_4):
            self.de_4_res.append(nn.Sequential(
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_4, dim_4, kernel_size=3, padding=1),
            self.activation))

        self.de_3_fuse = nn.Sequential(
            nn.Conv2d(dim_4 + dim_3, dim_3, kernel_size=3, padding=1),
            self.activation)
        self.de_3_res = nn.ModuleList()
        for i in range(self.number_de_3):
            self.de_3_res.append(nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation))

        self.de_2_fuse = nn.Sequential(
            nn.Conv2d(dim_3 + dim_2, dim_2, kernel_size=3, padding=1),
            self.activation)
        self.de_2_res = nn.ModuleList()
        for i in range(self.number_de_2):
            self.de_2_res.append(nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation))

        self.de_1_fuse = nn.Sequential(
            nn.Conv2d(dim_2 + dim_1, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.de_1_res = nn.ModuleList()
        for i in range(self.number_de_1):
            self.de_1_res.append(nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation))

        self.output = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
            self.activation)

    def forward(self, x, res_1, res_2, res_3):

        for i in range(self.number_de_4):
            x = self.activation(self.de_4_res[i](x) + x)

        hx = self.up(x)
        hx = self.de_3_fuse(torch.cat((hx, res_3), dim=1))
        for i in range(self.number_de_3):
            hx = self.activation(self.de_3_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_2_fuse(torch.cat((hx, res_2), dim=1))
        for i in range(self.number_de_2):
            hx = self.activation(self.de_2_res[i](hx) + hx)

        hx = self.up(hx)
        hx = self.de_1_fuse(torch.cat((hx, res_1), dim=1))
        for i in range(self.number_de_1):
            hx = self.activation(self.de_1_res[i](hx) + hx)

        output = self.output(hx)

        return output

class PHDT(nn.Module):
    def __init__(self, dim_1=16, dim_2=32, dim_3=64, dim_4=128):
        super(PHDT, self).__init__()

        self.encoder_A = Encoder_A(dim_1, dim_2, dim_3, dim_4)
        self.encoder_T = Encoder_T(dim_1, dim_2, dim_3, dim_4)
        self.encoder_J = Encoder_J(dim_1, dim_2, dim_3, dim_4)
        self.decoder = Decoder(dim_1, dim_2, dim_3, dim_4)

    def forward(self, haze, clean):

        A = self.encoder_A(haze)  # 256  0-1
        T = self.encoder_T(haze)  # h/32, w/32, 256 0-1
        J, res_1, res_2, res_3 = self.encoder_J(clean)  # h/32, w/32, 256

        F_x = J * T + A * (1 - T)

        rehaze = self.decoder(F_x, res_1, res_2, res_3)

        return rehaze

class PHATNet(nn.Module):
    def __init__(self):
        super(PHATNet, self).__init__()
        self.PHDT_1 = PHDT()
        self.PHDT_2 = PHDT()
        self.PHDT_3 = PHDT()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, haze, clean):

        haze_down_1 = F.interpolate(haze, scale_factor=0.5, mode='bilinear')
        haze_down_2 = F.interpolate(haze, scale_factor=0.25, mode='bilinear')
        clean_down_1 = F.interpolate(clean, scale_factor=0.5, mode='bilinear')
        clean_down_2 = F.interpolate(clean, scale_factor=0.25, mode='bilinear')

        rehaze_2 = self.PHDT_1(haze_down_2, clean_down_2).clamp(0, 1)
        rehaze_1 = (self.PHDT_2(haze_down_1, clean_down_1) + self.up(rehaze_2)).clamp(0, 1)
        rehaze_0 = (self.PHDT_3(haze, clean) + self.up(rehaze_1)).clamp(0, 1)

        return rehaze_0, rehaze_1, rehaze_2


if __name__ == '__main__':
    # Debug
    import time
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = PHATNet().cuda()
    haze = torch.randn(1, 3, 256, 256).cuda()
    clean = torch.randn(1, 3, 256, 256).cuda()

    flops, params = profile(net, (haze, clean))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')