# utilization methods like the discord bot, noise generator, upsampling things, load and restore functions
from utils.imresize import *
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters

import os
import random
from sklearn.cluster import KMeans


def generate_noise(size, device, num_samp=1, type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(
            size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(
            num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(
            num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)],
                    mode='bilinear', align_corners=True)
    return m(im)


def creat_reals_pyramid(real, reals, scale_factor, stop_scale):
    real = real[:, 0:3, :, :]
    for i in range(0, stop_scale + 1, 1):
        scale = math.pow(scale_factor, stop_scale - i)
        curr_real = imresize(real, scale)
        reals.append(curr_real)
    return reals


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp, 0, 1)
    return inp


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def np2torch(x, nc_im):
    if nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x


def read_image(input_dir, input_name):
    x = img.imread('%s/%s' % (input_dir, input_name))
    x = np2torch(x, 3)
    x = x[:, 0:3, :, :]
    return x


def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t


def adjust_scales2image(real_, min_size, scale_factor_init, max_size):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    num_scales = math.ceil((math.log(math.pow(
        min_size / (min(real_.shape[2], real_.shape[3])), 1), scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([max_size, max(
        [real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), scale_factor_init))
    stop_scale = num_scales - scale2stop
    # min(250/max([real_.shape[0],real_.shape[1]]),1)
    scale1 = min(max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, scale1)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    scale_factor = math.pow(
        min_size / (min(real.shape[2], real.shape[3])), 1 / (stop_scale))
    scale2stop = math.ceil(math.log(min([max_size, max(
        [real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), scale_factor_init))
    stop_scale = num_scales - scale2stop
    return real, stop_scale, scale_factor


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model
