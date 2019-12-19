import sys
import shutil
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
from random_samples import random_samples
import fastrcnn_mask_functions
import matplotlib.pyplot as plt

def handle_new_output_dir(opt):
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.delete_previous:
            shutil.rmtree(dir2save)
        elif opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' %
                  (opt.input_name, opt.gen_start_scale))
            exit(1)
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' %
                  (opt.input_name, opt.scale_h, opt.scale_v))
            exit(1)
        elif opt.mode == 'object_completion':
            print('object_completion results for image %s, start scale=%d, already exist' %
                  (opt.input_name, opt.gen_start_scale))
            exit(1)

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

def semantic_segmentation(minibatch: torch.Tensor, opt):
    # We need two classes only - background and person
    num_classes = 2
    # get the model using our helper function
    model = fastrcnn_mask_functions.get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(opt.device)
    checkpoint = torch.load(opt.sem_seg_model, map_location=opt.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        prediction = model(minibatch)
    assert len(prediction[0]['masks']) != 0, "No segmentation masks found! Abort."
    masks = prediction[0]['masks']
    masks[masks<=opt.mask_threshold] = 0
    masks[masks> opt.mask_threshold] = 1

    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t = masks[i]
        mask_t = mask_t[None,:,:,:]
        plt.imsave('%s/mask%s.png' % (dir2save, i), functions.convert_image_np(mask_t.detach()), vmin=0, vmax=1)

    return masks

def shift_masks(masks: torch.Tensor, opt):
    trans_v = opt.v_translation
    trans_h = opt.h_translation

    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t = masks[i].squeeze()

        ind = (mask_t==1).nonzero()
        ind_v   = ind[:,0] + trans_v
        ind_h   = ind[:,1] + trans_h
        ind_v[ind_v>=mask_t.shape[0]] = mask_t.shape[0]-1
        ind_h[ind_h>=mask_t.shape[1]] = mask_t.shape[1]-1

        shifted_mask_t = torch.zeros_like(mask_t)
        shifted_mask_t[ind_v, ind_h] = 1

        plt.imsave('%s/mask_s%s.png' % (dir2save, i),
                   functions.convert_image_np(shifted_mask_t[None,None,:,:].detach()), vmin=0, vmax=1)

        if i == 0:
            shifted_masks = shifted_mask_t[None,None,:,:]
        else:
            shifted_masks = torch.cat((shifted_masks, shifted_mask_t[None,None]),dim=0)

    return shifted_masks

def apply_segmentation_patch(minibatch: torch.Tensor, masks: torch.Tensor, masks_s: torch.Tensor, opt):
    assert isinstance(minibatch, torch.Tensor), "Mini-Batch of generated images most be an instance of torch.Tensor."
    assert isinstance(masks, torch.Tensor), "Semantic masks based on generated images most be an" \
                                            "instance of torch.Tensor."
    assert isinstance(masks_s, torch.Tensor), "Semantic masks based on generated images most" \
                                                    "be an instance of torch.Tensor."
    assert (masks.shape[0] == minibatch.shape[0]) and \
           (masks.shape[2] == minibatch.shape[2]) and \
           (masks.shape[3] == minibatch.shape[3]) and \
           (masks_s.shape[0] == minibatch.shape[0]) and \
           (masks_s.shape[2] == minibatch.shape[2]) and \
           (masks_s.shape[3] == minibatch.shape[3]), "Semantic masks and minibatch most have the same dimensions."

    masked_minibatch = masks * minibatch
    shifted_masked_minibatch = masks_s * minibatch

    dir2save = functions.generate_dir2save(opt)
    for i in range(masked_minibatch.shape[0]):
        masked_t = masked_minibatch[i]
        masked_t = masked_t[None,:,:,:]
        plt.imsave(('%s/masked%s.png' % (dir2save, i)),functions.convert_image_np(masked_t.detach()), vmin=0, vmax=1)
        masked_t = shifted_masked_minibatch[i]
        masked_t = masked_t[None, :, :, :]
        plt.imsave(('%s/masked_s%s.png' % (dir2save, i)), functions.convert_image_np(masked_t.detach()), vmin=0, vmax=1)

    return masked_minibatch, shifted_masked_minibatch

def replace_patches(minibatch: torch.Tensor, masks: torch.Tensor, masks_s: torch.Tensor, opt):
    assert isinstance(minibatch, torch.Tensor), "Mini-Batch of generated images most be an instance of torch.Tensor."
    assert isinstance(masks, torch.Tensor), "Semantic masks based on generated images most be an" \
                                            "instance of torch.Tensor."
    assert isinstance(masks_s, torch.Tensor), "Semantic masks based on generated images most" \
                                                    "be an instance of torch.Tensor."
    assert (masks.shape[0] == minibatch.shape[0]) and \
           (masks.shape[2] == minibatch.shape[2]) and \
           (masks.shape[3] == minibatch.shape[3]) and \
           (masks_s.shape[0] == minibatch.shape[0]) and \
           (masks_s.shape[2] == minibatch.shape[2]) and \
           (masks_s.shape[3] == minibatch.shape[3]), "Semantic masks and minibatch most have the same dimensions."

    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t1 = masks[i].squeeze()
        mask_t2 = masks_s[i].squeeze()
        sample  = minibatch[i].clone().squeeze()

        sample_r = sample[0,:,:].squeeze()
        sample_g = sample[1,:,:].squeeze()
        sample_b = sample[2,:,:].squeeze()

        sample_r[mask_t1==1]=sample_r[mask_t2==1]
        sample_g[mask_t1==1]=sample_g[mask_t2==1]
        sample_b[mask_t1==1]=sample_b[mask_t2==1]

        sample = torch.cat((sample_r[None,:,:], sample_g[None,:,:], sample_b[None,:,:]), dim=0)

        plt.imsave('%s/replace%s.png' % (dir2save, i),
                   functions.convert_image_np(sample[None,:,:,:].detach()), vmin=0, vmax=1)

        if i == 0:
            replaced_samples = sample[None,:,:,:]
        else:
            replaced_samples = torch.cat((replaced_samples, sample[None,:,:,:]), dim=0)

    return replaced_samples

def create_masks_pyramids(masks: torch.Tensor, opt, shifted=False):
    pyramids = {}
    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t = masks[i]
        pyramid = []
        pyramid = functions.creat_reals_pyramid(mask_t[None,:,:,:],pyramid,opt)
        for mask in pyramid:
            unique, count = torch.unique(mask, return_counts=True, sorted=True)
            majority = unique[torch.argmax(count)]
            mask[mask == majority] = 0
            mask[mask != 0] = 1
            # functions.plot_minibatch(mask, f'{mask.shape}', opt)

        torch.save(pyramid, '{}/{}_pyramid.pth'.format(dir2save, f'mask{i}' if not shifted else f'mask_s{i}'))
        pyramids[i] = pyramid
    return pyramids

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples',   default=1)


    opt = parser.parse_args()
    opt = functions.post_config(opt)

    assert opt.mode == 'object_completion' and opt.sem_seg_model != None, "In, 'object_completion' mode, one most provide" \
                                                                      "semantic segmentation model name."
    assert os.path.exists(os.path.join(opt.sem_seg_dir, opt.sem_seg_model)), "Semantic Segmentation model is not exist!"

    opt.sem_seg_model = os.path.join(opt.sem_seg_dir, opt.sem_seg_model)

    handle_new_output_dir(opt)

    dir2save = functions.generate_dir2save(opt)
    real_    = functions.read_image(opt)
    functions.adjust_scales2image(real_, opt)
    plt.imsave('%s/real.png' % (dir2save), functions.convert_image_np(real_.detach()), vmin=0, vmax=1)

    masks_   = semantic_segmentation(real_, opt)
    masks_s_ = shift_masks(masks_, opt)
    masked_, masked_s_ = apply_segmentation_patch(minibatch=real_, masks=masks_, masks_s=masks_s_, opt=opt)
    replace_ = replace_patches(real_, masks_, masks_s_, opt)
    if opt.plotting:
        functions.plot_minibatch(torch.cat((real_,
                                            torch.cat((masks_,masks_,masks_), dim=1),
                                            masked_,
                                            torch.cat((masks_s_,masks_s_,masks_s_), dim=1),
                                            masked_s_,
                                            replace_), dim=0),
                                 f'shape={real_.shape}', opt)

    masks   = imresize(masks_,opt.scale1,opt)
    masks_s = imresize(masks_s_,opt.scale1,opt)

    masks_pyramid   = create_masks_pyramids(masks,opt)
    # masks_s_pyramid = create_masks_pyramids(masks_s,opt,shifted=True)
    Modified = random_samples(opt, mask_pyramid=masks_pyramid)

    if opt.plotting:
        functions.plot_minibatch(Modified, f'G(Z0...Z{opt.gen_start_scale}), shape={Modified.shape}\n'
                                           f'(After Latent space arithmetic)', opt)

