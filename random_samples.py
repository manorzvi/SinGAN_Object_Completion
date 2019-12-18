from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import shutil
import os

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

def manipulate_noise(Z_pyramid: list, mask_pyramid: dict, m_shift_pyramid: dict):
    assert (len(Z_pyramid) == len(mask_pyramid)) and \
           (len(Z_pyramid) == len(m_shift_pyramid)), "Mask pyramids must have the same No of scales as the" \
                                                     "noise pyramid."
    for z, m, m_shifted in zip(Z_pyramid, mask_pyramid, m_shift_pyramid):
        assert ((z.shape[2] == m.shape[2]) and (z.shape[3] == m.shape[3])) and \
               ((z.shape[2] == m_shifted.shape[2]) and (z.shape[3] == m_shifted.shape[3])), \
            'All masks must have the same size as the noise.'

    for i in range(len(Z_pyramid)):
        Z_pyramid[i] = manipulate_single_scale(Z_pyramid[i], mask_pyramid[i], m_shift_pyramid[i])

    return Z_pyramid

def random_samples(opt, mask_pyramid=None, shifted_mask_pyramid=None):
    assert (not mask_pyramid and not shifted_mask_pyramid) or \
           (isinstance(mask_pyramid,dict) and isinstance(shifted_mask_pyramid,dict)), "If applied, both pyramids should" \
                                                                                      " be valid."
    if shifted_mask_pyramid and shifted_mask_pyramid:
        assert len(mask_pyramid) == len(shifted_mask_pyramid), "If applied, both pyramids should have the same length."
        for (i1,m1), (i2,m2) in zip(mask_pyramid.items(),shifted_mask_pyramid.items()):
            for m11, m21 in zip(m1,m2):
                assert m11.shape == m21.shape, "If applied mask pyramids, all masks should have the same sizes."

    Gs       = []
    Zs       = []
    reals    = []
    NoiseAmp = []

    if opt.save_noise_pyramid:
        Ns = {n: [] for n in range(opt.num_samples)}

    if not opt.pyramid:
        handle_new_output_dir(opt)

    if opt.mode == 'random_samples':
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        in_s = functions.generate_in2coarsest(reals, 1, 1, opt)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt,
                        gen_start_scale=opt.gen_start_scale, num_samples=opt.num_samples,
                        Ns=(Ns if opt.save_noise_pyramid else None))

    elif opt.mode == 'random_samples_arbitrary_sizes':
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        in_s = functions.generate_in2coarsest(reals, opt.scale_v, opt.scale_h, opt)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h,
                        num_samples=opt.num_samples,
                        Ns=(Ns if opt.save_noise_pyramid else None))

    elif opt.mode == 'object_completion':
        real = functions.read_image(opt)
        if opt.plotting and not opt.pyramid:
            functions.plot_minibatch(real, f'Original Real, shape={real.shape}', opt)
        functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        in_s = functions.generate_in2coarsest(reals, 1, 1, opt)
        return SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt,
                               gen_start_scale=opt.gen_start_scale, num_samples=opt.num_samples,
                               Ns=(Ns if opt.save_noise_pyramid else None),
                               mask_pyramid=(mask_pyramid if opt.pyramid else None),
                               shifted_mask_pyramid=(shifted_mask_pyramid if opt.pyramid else None))

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train',required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    random_samples(opt)





