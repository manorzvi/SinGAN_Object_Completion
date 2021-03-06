from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments
from pprint import pprint

def generate_gif(Gs,Zs,reals,NoiseAmp,opt,alpha=0.1,beta=0.9,start_scale=2,fps=10):

    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G,Z_opt,noise_amp,real in zip(Gs,Zs,NoiseAmp,reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        #pad_noise = 0
        #m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
            z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
            z_prev1 = 0.95*Z_opt +0.05*z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0,100,1):
            if count == 0:
                z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_rand = z_rand.expand(1,3,Z_opt.shape[2],Z_opt.shape[3])
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
            else:
                diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device))

            z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                #I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp*z_curr+I_prev
            I_curr = G(z_in.detach(),I_prev)

            if (count == len(Gs)-1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0,:,:,:].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0)*255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save,start_scale,alpha,beta),images_cur,fps=fps)
    del images_cur

def manipulate_single_scale(Z: torch.Tensor, mask: torch.Tensor, m_shift: torch.Tensor):
    m_t = mask.squeeze()
    m_shift_t = m_shift.squeeze()
    Z_r = Z.squeeze()[0,:,:].squeeze()
    Z_g = Z.squeeze()[1,:,:].squeeze()
    Z_b = Z.squeeze()[2,:,:].squeeze()
    Z_r[m_t == 1] = Z_r[m_shift_t == 1]
    Z_g[m_t == 1] = Z_g[m_shift_t == 1]
    Z_b[m_t == 1] = Z_b[m_shift_t == 1]

    # Z_r[m_t == 1] = 0
    # Z_g[m_t == 1] = 0
    # Z_b[m_t == 1] = 0

    Z = torch.cat((Z_r[None, :, :], Z_g[None, :, :], Z_b[None, :, :]), dim=0)[None,:,:,:]
    return Z

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,
                    opt,in_s=None,scale_v=1,scale_h=1,n=0,
                    gen_start_scale=0,num_samples=50, Ns=None, mask_pyramid=None, shifted_mask_pyramid=None):
    # TODO: BUG. trained relatively large image with max_size=1024, min_size=32 (on chinese_woman.jpg),
    #  and got 12 scales for reals and only 10 scales for Gs,Zs,NoiseAmp
    print(f'[debug] - |reals|={len(reals)} , |Gs|={len(Gs)}')
    assert (opt.save_noise_pyramid and Ns is not None) or not opt.save_noise_pyramid, "if save_noise_pyramid " \
                                                                                      "option is active, " \
                                                                                      "you must provide Ns - " \
                                                                                      "a nested dictionary to save " \
                                                                                      "the intermediate noises."
    if opt.save_noise_pyramid:
        assert isinstance(Ns, dict)
    # TODO: review. init Generated to None.
    #  later on would be populated by generated images. (manorz, 12/18/19)
    if opt.mode == 'object_completion':
        Generated = None
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)

    images_cur = []

    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0: # Single-channel noise map only in the first scale
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else: # Number of channels as in original image.
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt
                # TODO: Ask Tamar: why Zs is all zeros except the first scale? (manorz, 12/18/19)
                # functions.plot_minibatch(z_curr,f'DEBUG (REMOVE LATER)\n'
                #                                 f'z_crr\n'
                #                                 f'(scale={n}, image={i})', opt) #TODO: remove later (manorz, 12/18/19)

            if opt.pyramid and n == 0:
                z_curr = manipulate_single_scale(z_curr, mask_pyramid[i][n], shifted_mask_pyramid[i][n])
                # functions.plot_minibatch(z_curr,f'DEBUG (REMOVE LATER)\n'
                #                                 f'z_crr after mask\n'
                #                                 f'(scale={n}, image={i})',opt) # TODO: remove later (manorz, 12/18/19)

            z_in = noise_amp*(z_curr)+I_prev
            I_curr = G(z_in.detach(),I_prev)

            # TODO: for each sample (i), and for each scale (n),
            #  save input noise map at Ns: Ns[sample idx][scale idx] (manorz, 12/13/19)
            if opt.save_noise_pyramid:
                Ns[i].append(z_curr)

            if n == len(reals)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & \
                        (opt.mode != "SR") & (opt.mode != "paint2image") & (opt.mode != 'object_completion'):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                    #plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    #plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
                if opt.mode == 'object_completion':
                    if not opt.pyramid:
                        plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,
                                   vmax=1)
                    else:
                        plt.imsave('%s/%d.modify.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0,
                                   vmax=1)
                # TODO: review create a mini-batch of generated images. (manorz, 12/18/19)
                if   (opt.mode == 'object_completion') and not isinstance(Generated, torch.Tensor):
                    Generated = I_curr.detach()
                elif opt.mode == 'object_completion':
                    Generated = torch.cat((Generated,I_curr), dim=0)
            images_cur.append(I_curr)
        n+=1

    if opt.save_noise_pyramid:
        dir2save = functions.generate_dir2save(opt)
        file2save = os.path.join(dir2save,'noise_pyramids.pth')
        torch.save(Ns, file2save)

    # TODO: review. only in 'object_detection' mode we need to return the generated images.
    #  Elsewhere we only need to save them. (manorz, 12/18/19)
    if opt.mode == 'object_completion':
        return Generated.detach()