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
import matplotlib.pyplot as plt

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
                   functions.convert_image_np(sample[None,:,:,:].detach()), vmin=0, vmax=1)

        if i == 0:
            replaced_samples = sample[None,:,:,:]
        else:
            replaced_samples = torch.cat((replaced_samples, sample[None,:,:,:]), dim=0)

    if opt.plotting:
        functions.plot_minibatch(replaced_samples,
                                 f'REPLACED(G(Z0...Z{opt.gen_start_scale})))), '
                                 f'shape={replaced_samples.shape}', opt)

    return replaced_samples

def plot_mask_pyramid(pyramid: list):
    t = int(np.ceil(np.sqrt(len(pyramid))))
    fig, axes = plt.subplots(t,t)
    fig.tight_layout()
    for i,mask in enumerate(pyramid):
        axes[int(i/t), int(i%t)].imshow(np.transpose(mask.squeeze().cpu().numpy(), (1,2,0)))
        axes[int(i/t), int(i%t)].set_title(str(mask.shape))

    plt.show()

def create_random_generated_masks_pyramid(masks: torch.Tensor, opt):
    pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
    m = nn.ZeroPad2d(int(pad1))
    pyramids = {}
    dir2save = functions.generate_dir2save(opt)

    for j in range(masks.shape[0]):
        mask_t = masks[j].squeeze()
        pyramid = []
        for i in range(0, opt.stop_scale + 1, 1):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_mask = imresize(mask_t[None,None,:,:], scale, opt)
            curr_mask = m(curr_mask)
            pyramid.append(curr_mask)
        torch.save(pyramid, ('%s/%d.mask_pyramid.pth' % (dir2save, j)))
        pyramids[j] = pyramid

        # plot_mask_pyramid(pyramid)

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

    opt.pyramid = False
    Generated = random_samples(opt)

    if opt.plotting:
        functions.plot_minibatch(Generated, f'G(Z0...Z{opt.gen_start_scale}), shape={Generated.shape}', opt)

    masks                  = semantic_segmentation(Generated, opt)
    shifted_masks          = shift_masks(masks, opt)
    masked, shifted_masked = apply_segmentation_patch(minibatch=Generated, masks=masks,
                                                      shifted_masks=shifted_masks, opt=opt)
    replace_patches(Generated, masks, shifted_masks, opt)

    pyramids         = create_random_generated_masks_pyramid(masks,opt)
    shifted_pyramids = create_random_generated_masks_pyramid(shifted_masks,opt)

    opt.pyramid = True
    Modified = random_samples(opt, mask_pyramid=pyramids, shifted_mask_pyramid=shifted_pyramids)

    if opt.plotting:
        functions.plot_minibatch(Modified, f'G(Z0...Z{opt.gen_start_scale}), shape={Modified.shape}\n'
                                           f'(After Latent space arithmetic)', opt)

