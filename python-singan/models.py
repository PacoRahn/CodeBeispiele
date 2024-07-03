<<<<<<< HEAD
# this file should contain the models architectures
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel,
                                          kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self, nc_im, ker_size, padd_size, nfc, num_layer, min_nfc):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(nfc)
        self.head = ConvBlock(nc_im, N, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc),
                              max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(
            max(N, min_nfc), 1, kernel_size=ker_size, stride=1, padding=padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, nc_im, ker_size, padd_size, nfc, num_layer, min_nfc):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = nfc
        # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.head = ConvBlock(nc_im, N, ker_size, padd_size, 1)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc),
                              max(N, min_nfc), ker_size, padd_size, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N, min_nfc), nc_im, kernel_size=ker_size,
                      stride=1, padding=padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y
=======
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
from torch.nn.modules import padding
from torch.nn.modules.conv import Conv2d



class customConv(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_size):
        self._is_cuda = torch.cuda.is_available()
        super().__init__()
        #need to find out whether strings are relevant
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        #inplace makes the data be changed without reassigning
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))



class Discriminator(nn.module):
    """[Allows initialization of a Discriminator for the SinGAN, following the proposed layer architecture
    For ease of use, only required initialization parameter is scaleLevel: int describing the current Pyramid level in SinGAN]
    """
    def __init__(self, scaleLevel, kernel_size=3, stride=1, padding_size=0):
        super().__init__()

        # every 4 scales we double number of kernels to a max of 128 like in GitHub
        inChannel, outChannel = 32 * pow(2, math.floor(scaleLevel / 4), 128)
        #different to GitHub we use functional programming instead of nn.Sequential
        
        #first parameter = 3 because we have RGB images
        self._l1 = customConv(3, outChannel, kernel_size, stride, padding_size)
        self._l2 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l3 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l4 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l5 = nn.Conv2d(inChannel, 1, kernel_size, stride, padding_size)
        

    def forward(self, x):
        x = self._l1(x)
        x = self._l2(x)
        x = self._l3(x)
        x = self._l4(x)
        discrimination = self._l5(x)
        return discrimination



class Generator(nn.module):
    def __init__(self, scaleLevel, kernel_size=3, stride=1, padding_size=0):
        super().__init__()

        inChannel, outChannel = 32 * pow(2, math.floor(scaleLevel / 4), 128)
        self._l1 = customConv(3, outChannel, kernel_size, stride, padding_size)
        self._l2 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l3 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l4 = customConv(inChannel, outChannel, kernel_size, stride, padding_size)
        self._l5 = Conv2d(inChannel, 3, kernel_size, stride, padding_size)

    def foward(self,x,y):
        #we suppose that x is noise and y is upsampled image
        if x.shape[2] != y.shape[2] or x.shape[3] != y.shape[3]:
            print("Image and Noise have different height and width")
            print("Starting debugging at current position")
            import ipdb; ipdb.set_trace()
        #currently we passing as mentioned in paper
        #different to github, where concat x+y is skipped
        combined = x
        combined = self._l1(combined)
        combined = self._l2(combined)
        combined = self._l3(combined)
        combined = self._l4(combined)
        combined = self._l5(combined)
        combined = nn.Tanh(combined)
        generatedImage = combined + y
        return generatedImage

>>>>>>> f76ce6c74479eccc81025ba4b94cfc76ea279cd3
