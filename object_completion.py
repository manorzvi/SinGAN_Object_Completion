import sys
import shutil
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
from random_samples import random_samples
import fastrcnn_mask_functions

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

    masks = prediction[0]['masks']
    for i in range(1, len(prediction)):
        masks = torch.cat((masks,prediction[i]['masks']), dim=0)

    masks[masks<=opt.mask_threshold] = 0
    masks[masks> opt.mask_threshold] = 1

    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t = masks[i]
        mask_t = mask_t[None,:,:,:]
        plt.imsave('%s/%d.mask1.png' % (dir2save, i), functions.convert_image_np(mask_t.detach()), vmin=0, vmax=1)

    if opt.plotting:
        functions.plot_minibatch(masks, f'MASK(G(Z0...Z{opt.gen_start_scale})), shape={masks.shape}', opt)

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

        plt.imsave('%s/%d.mask3.png' % (dir2save, i),
                   functions.convert_image_np(shifted_mask_t[None,None,:,:].detach()), vmin=0, vmax=1)

        if i == 0:
            shifted_masks = shifted_mask_t[None,None,:,:]
        else:
            shifted_masks = torch.cat((shifted_masks, shifted_mask_t[None,None]),dim=0)

    if opt.plotting:
        functions.plot_minibatch(shifted_masks,
                                 f'SHIFT(MASK(G(Z0...Z{opt.gen_start_scale})))), shape={shifted_masks.shape}', opt)
    return shifted_masks


def apply_segmentation_patch(minibatch: torch.Tensor, masks: torch.Tensor, shifted_masks: torch.Tensor, opt):
    assert isinstance(minibatch, torch.Tensor), "Mini-Batch of generated images most be an instance of torch.Tensor."
    assert isinstance(masks, torch.Tensor), "Semantic masks based on generated images most be an" \
                                            "instance of torch.Tensor."
    assert isinstance(shifted_masks, torch.Tensor), "Semantic masks based on generated images most" \
                                                    "be an instance of torch.Tensor."
    assert (masks.shape[0] == minibatch.shape[0]) and \
           (masks.shape[2] == minibatch.shape[2]) and \
           (masks.shape[3] == minibatch.shape[3]) and \
           (shifted_masks.shape[0] == minibatch.shape[0]) and \
           (shifted_masks.shape[2] == minibatch.shape[2]) and \
           (shifted_masks.shape[3] == minibatch.shape[3]), "Semantic masks and minibatch most have the same dimensions."

    masked_minibatch = masks * minibatch
    shifted_masked_minibatch = shifted_masks * minibatch

    dir2save = functions.generate_dir2save(opt)
    for i in range(masked_minibatch.shape[0]):
        masked_t = masked_minibatch[i]
        masked_t = masked_t[None,:,:,:]
        plt.imsave(('%s/%d.mask2.png' % (dir2save, i)),functions.convert_image_np(masked_t.detach()), vmin=0, vmax=1)
        masked_t = shifted_masked_minibatch[i]
        masked_t = masked_t[None, :, :, :]
        plt.imsave(('%s/%d.mask4.png' % (dir2save, i)), functions.convert_image_np(masked_t.detach()), vmin=0, vmax=1)

    if opt.plotting:
        functions.plot_minibatch(masked_minibatch,
                                 f'MASK(G(Z0...Z{opt.gen_start_scale}).*G(Z0...Z{opt.gen_start_scale})),'
                                 f'shape={masked_minibatch.shape}', opt)
        functions.plot_minibatch(shifted_masked_minibatch,
                                 f'SHIFT(MASK(G(Z0...Z{opt.gen_start_scale})).*G(Z0...Z{opt.gen_start_scale})),'
                                 f'shape={shifted_masked_minibatch.shape}', opt)

    return masked_minibatch, shifted_masked_minibatch


def replace_patches(minibatch: torch.Tensor, masks: torch.Tensor, shifted_masks: torch.Tensor, opt):
    assert isinstance(minibatch, torch.Tensor), "Mini-Batch of generated images most be an instance of torch.Tensor."
    assert isinstance(masks, torch.Tensor), "Semantic masks based on generated images most be an" \
                                            "instance of torch.Tensor."
    assert isinstance(shifted_masks, torch.Tensor), "Semantic masks based on generated images most" \
                                                    "be an instance of torch.Tensor."
    assert (masks.shape[0] == minibatch.shape[0]) and \
           (masks.shape[2] == minibatch.shape[2]) and \
           (masks.shape[3] == minibatch.shape[3]) and \
           (shifted_masks.shape[0] == minibatch.shape[0]) and \
           (shifted_masks.shape[2] == minibatch.shape[2]) and \
           (shifted_masks.shape[3] == minibatch.shape[3]), "Semantic masks and minibatch most have the same dimensions."

    dir2save = functions.generate_dir2save(opt)
    for i in range(masks.shape[0]):
        mask_t1 = masks[i].squeeze()
        mask_t2 = shifted_masks[i].squeeze()
        sample  = minibatch[i].squeeze()

        sample_r = sample[0,:,:].squeeze()
        sample_g = sample[1,:,:].squeeze()
        sample_b = sample[2,:,:].squeeze()

        sample_r[mask_t1==1]=sample_r[mask_t2==1]
        sample_g[mask_t1==1]=sample_g[mask_t2==1]
        sample_b[mask_t1==1]=sample_b[mask_t2==1]

        sample = torch.cat((sample_r[None,:,:], sample_g[None,:,:], sample_b[None,:,:]), dim=0)

        plt.imsave('%s/%d.mask5.png' % (dir2save, i),
                   functions.convert_image_np(sample[None, :, :].detach()), vmin=0, vmax=1)

def creat_random_generated_masks_pyramid(masks: torch.Tensor):
    pass

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
    parser.add_argument('--delete_previous', action='store_true', help='delete previous results directory, if exists.', default=False)
    parser.add_argument('--save_noise_pyramid', action='store_true', help='for each sample, save its noise pyramid.',      default=False)
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate at each run of random_samples.py')
    parser.add_argument('--plotting', action='store_true', default=False, help='Plot images at selected points (which is important to examine to process)')
    parser.add_argument('--sem_seg_dir', default='models', help='Semantic segmentation trained models directory')
    parser.add_argument('--sem_seg_model', help='Semantic segmentation trained model.')
    parser.add_argument('--mask_threshold', default=0.01, help='Threshold for segmentation masking')
    parser.add_argument('--v_translation', '-Tv', type=int, default=10, help='Segmentation mask vertical translation.')
    parser.add_argument('--h_translation', '-Th', type=int, default=10, help='Segmentation mask horizontal translation.')

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    assert opt.mode == 'object_completion' and opt.sem_seg_model != None, "In, 'object_completion' mode, one most provide" \
                                                                      "semantic segmentation model name."
    assert os.path.exists(os.path.join(opt.sem_seg_dir, opt.sem_seg_model)), "Semantic Segmentation model is not exist!"

    opt.sem_seg_model = os.path.join(opt.sem_seg_dir, opt.sem_seg_model)

    Generated = random_samples(opt)
    if opt.plotting:
        functions.plot_minibatch(Generated, f'G(Z0...Z{opt.gen_start_scale}), shape={Generated.shape}', opt)

    masks                  = semantic_segmentation(Generated, opt)
    shifted_masks          = shift_masks(masks, opt)
    masked, shifted_masked = apply_segmentation_patch(minibatch=Generated, masks=masks,
                                                      shifted_masks=shifted_masks, opt=opt)
    replace_patches(Generated, masks, shifted_masks, opt)

    pyramid = creat_random_generated_masks_pyramid(masks,opt)



