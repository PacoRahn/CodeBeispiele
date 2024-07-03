import utils.utils as util
from typing import List
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt


import Architecture.models as models
from utils.imresize import imresize

# Temp variables for future assigment:
GeneratorType = any
DiscriminatorType = any


def train_pyramid(G_pyramid, Noise_pyramid, input_path, input_name, NoiseAmp, scale_stop=4, scale_factor=0.75, gpu_id='cuda:0'):
    """Fully train all Generator-Discriminator pairs
    """
    input_full_name = input_name + '.png'
    real_ = util.read_image(input_path, input_full_name)
    reals = []
    in_s = 0
    scale_num = 0
    real = imresize(real_, scale_factor)
    reals = util.creat_reals_pyramid(
        real, reals, scale_factor, scale_stop)

    device = torch.device(gpu_id)

    while scale_num < scale_stop + 1:
        print("start trainig on scale: " + str(scale_num + 1))
        scaleLevel = scale_num + 1
        D_curr, G_curr = init_models(device)

        noise_curr, in_s, G_curr = train_on_single_scale(
            G_curr, D_curr, G_pyramid, Noise_pyramid, reals, NoiseAmp, in_s, input_name)

        G_curr = util.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = util.reset_grads(D_curr, False)
        D_curr.eval()

        G_pyramid.append(G_curr)
        Noise_pyramid.append(noise_curr)
        NoiseAmp.append(0.1)

        scale_num += 1
        del G_curr, D_curr
        #print("Trained on scale: " + str(int(scale_num + 1)))
        #print('in_s: ' + str(in_s))
    return G_pyramid, Noise_pyramid

# functions for training the model


def train_on_single_scale(G_n, D_n, G_pyramid, Z_pyramid, reals, NoiseAmp, in_s, input_name, d_steps=3, g_steps=3, lr=0.0005, num_epochs=2000, gpu_id="cuda:0", lambda_grad=0.1, noise_num_channel=3, noise_amp_init=0.1, scale_factor=0.75, alpha=10, beta1=0.5):
    # TODO implement training on single scale test
    #print("Pytorch version is:")
    # print(torch.__version__)
    real = reals[len(G_pyramid)]
    print('G_pyramid now has length: ' + str(len(G_pyramid)))
    scale_level = len(G_pyramid) + 1
    nx = real.shape[2]  # why shape[2] and not 0?
    ny = real.shape[3]
    device = torch.device(gpu_id)

    # reconstruction loss weight
    alpha = alpha

    # Padding
    pad_noise = 5
    pad_image = 5
    padding_noise = nn.ZeroPad2d(pad_noise)
    padding_image = nn.ZeroPad2d(pad_image)

    # initial noise
    fixed_noise = util.generate_noise(
        [noise_num_channel, nx, ny], device=device)
    noise_opt = padding_noise(torch.full(fixed_noise.shape, 0, device=device))

    optimizerD = optim.Adam(D_n.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G_n.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerD, milestones=[1600], gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizerG, milestones=[1600], gamma=0.1)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(num_epochs):

        # Generate Noise
        if(G_pyramid == []):
            noise_opt = util.generate_noise([1, nx, ny], device=device)
            noise_opt = padding_noise(noise_opt.expand(1, 3, nx, ny))
            noise_ = util.generate_noise([1, nx, ny], device=device)
            noise_ = padding_noise(noise_.expand(1, 3, nx, ny))
        else:
            noise_ = util.generate_noise(
                [noise_num_channel, nx, ny], device=device)
            noise_ = padding_noise(noise_)

        ########################################
        # Update Discriminator network
        ########################################
        for i in range(d_steps):
            #############################
            # Train with all real images
            #############################
            D_n.zero_grad()

            output = D_n(real).to(device)

            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)

            #############################
            # Train with all fake images
            #############################

            # get generated images and noise from previous stage
            if(i == 0) & (epoch == 0):
                if(G_pyramid == []):
                    prev = torch.full(
                        [1, noise_num_channel, nx, ny], 0, device=device)
                    in_s = prev
                    prev = padding_image(prev)
                    z_prev = torch.full(
                        [1, noise_num_channel, nx, ny], 0, device=device)
                    z_prev = padding_noise(z_prev)
                    noise_amp = 1
                else:
                    prev = draw_concat(G_pyramid, Z_pyramid, reals, NoiseAmp, in_s,
                                       'rand', padding_noise, padding_image, gpu_id, scale_factor, noise_num_channel)
                    prev = padding_image(prev)
                    z_prev = draw_concat(G_pyramid, Z_pyramid, reals, NoiseAmp, in_s,
                                         'rec', padding_noise, padding_image, gpu_id, scale_factor, noise_num_channel)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    noise_amp = noise_amp_init * RMSE
                    z_prev = padding_noise(z_prev)

            else:
                prev = draw_concat(G_pyramid, Z_pyramid, reals, NoiseAmp, in_s,
                                   'rand', padding_noise, padding_image, gpu_id, scale_factor, noise_num_channel)
                prev = padding_image(prev)

            if (G_pyramid == []):
                G_input = noise_
            else:
                G_input = noise_amp * noise_ + prev

            fake = G_n(G_input.detach(), prev)

            output = D_n(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)

            gradient_penalty = calc_gradient_penalty(
                D_n, real, fake, lambda_grad, device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

    ########################################
    # Update Generator network
    ########################################
        for j in range(g_steps):
            torch.autograd.set_detect_anomaly(True)
            G_n.zero_grad()
            output = D_n(fake.clone())
            D_fake_map = output.detach()

            errG = -output.mean()
            #print("problem is in second G step")
            errG.backward(retain_graph=True)
            #print('G Backward successfully')

            if alpha != 0:
                loss = nn.MSELoss()
                new_noise_amp = noise_amp
                Z_opt = noise_amp * noise_opt + z_prev
                rec_loss = alpha * loss(G_n(Z_opt.detach(), prev), real)
                #print("problem is in reconstruction loss")
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = noise_opt
                rec_loss = 0
           # print("problem is in optimizer step")
            optimizerG.step()

        # if epoch % 500 == 0:

            #print("epoch: " + str(epoch))
        #print('finished one epoch')
        schedulerD.step()
        schedulerG.step()
    file_name = '%s/' + input_name + '_fake' + "_" + str(scale_level) + ".png"
    plt.imsave(file_name % './storage',
               util.convert_image_np(fake.detach()), vmin=0, vmax=1)
    return noise_opt, in_s, G_n


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    # print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)  # .cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(
                                        device),  # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, gpu_id, scale_factor, noise_num_channel):
    G_z = in_s
    device = torch.device(gpu_id)
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = 5
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = util.generate_noise(
                        [1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = util.generate_noise(
                        [noise_num_channel, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=device)
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / scale_factor)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / scale_factor)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                # if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z


def init_models(device, nc_im=3, ker_size=3, padd_size=0, nfc=32, num_layer=5, min_nfc=32):

    # generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(
        nc_im, ker_size, padd_size, nfc, num_layer, min_nfc).to(device)
    netG.apply(models.weights_init)

    # print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(
        nc_im, ker_size, padd_size, nfc, num_layer, min_nfc).to(device)
    netD.apply(models.weights_init)

    # print(netD)

    return netD, netG
