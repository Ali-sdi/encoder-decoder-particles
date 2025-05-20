
import torchvision.transforms as transform
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torch

import torch.nn as nn
import torch.nn.functional as functional


class Logs:
    def __init__(self, printlogs=False, pooling=False):
        global PRINTLOG, POOLING
        PRINTLOG = printlogs
        POOLING = pooling

    def __str__(self):
        return f"Printing Generator logs : {PRINTLOG}, Pooling logs : {POOLING}"


class Convblock(nn.Module):

    def __init__(self, input_channel, output_channel, kernal=3, stride=1, padding=1, upconve=True):

        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernal, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)


        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class Downconv(nn.Module):

    def __init__(self, input_channel, output_channel, kernal=4, stride=2, padding=1, upconve=True):

        super().__init__()

        self.upconvblock = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernal, stride, padding),

            nn.ReLU(inplace=True),


        )

    def forward(self, x):
        x = self.upconvblock(x)
        return x


class Upconv(nn.Module):

    def __init__(self, input_channel, output_channel, kernal=4, stride=2, padding=1, output_padding=0):
        super().__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel,
                               kernal, stride, padding, output_padding,),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upconv(x)
        return x


# ------------------------------------------------------

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x



class UNet(nn.Module):

    def __init__(self, input_channel=3, retain=True):

        super().__init__()
        self.conv1 = Convblock(input_channel, 32)
        self.dconv1 = Downconv(32, 32)
        self.conv2 = Convblock(32, 64)
        self.dconv2 = Downconv(64, 64)
        self.conv3 = Convblock(64, 128)
        self.dconv3 = Downconv(128, 128)
        self.conv4 = Convblock(128, 256)
        self.dconv4 = Downconv(256, 256)

        self.neck = nn.Conv2d(256, 512, 3, 1, 1)

        self.upconv4 = Upconv(512, 256, 4, 2, 1, 0)
        self.Dconv4 = Convblock(512, 256)
        self.upconv3 = Upconv(256, 128, 4, 2, 1, 0)
        self.Dconv3 = Convblock(256, 128)
        self.upconv2 = Upconv(128, 64, 4, 2, 1, 0)
        self.Dconv2 = Convblock(128, 64)
        self.upconv1 = Upconv(64, 32, 4, 2, 1, 0)
        self.Dconv1 = Convblock(64, 32)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        self.retain = retain

    def forward(self, x):

        conv1 = self.conv1(x)
        pool1 = self.dconv1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.dconv2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.dconv3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.dconv4(conv4)

        neck = self.neck(pool4)

        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4, upconv4)
        concat = torch.cat([upconv4, croped], 1)
        dconv4 = self.Dconv4(concat)

        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3, upconv3)
        concat = torch.cat([upconv3, croped], 1)
        dconv3 = self.Dconv3(concat)

        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2, upconv2)
        dconv2 = self.Dconv2(torch.cat([upconv2, croped], 1))

        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1, upconv1)
        dconv1 = self.Dconv1(torch.cat([upconv1, croped], 1))

        out = self.out(dconv1)
        if self.retain == True:
            out = functional.interpolate(out, list(x.shape)[2:])
        return functional.sigmoid(out)

    def crop(self, input_tensor, target_tensor):

        _, _, H, W = target_tensor.shape
        return transform.CenterCrop([H, W])(input_tensor)


if __name__ == "__main__":
    inputs = torch.randn((4, 3, 256, 256))
    model = build_resunet()
    y = model(inputs)
    print(y.shape)