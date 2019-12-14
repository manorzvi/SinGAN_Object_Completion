from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import shutil
from random_samples import random_samples

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int,
                        help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h',         type=float,
                        help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v',         type=float,
                        help='vertical resize factor for random samples',   default=1)

    parser.add_argument('--delete_previous',    action='store_true',
                        help='delete previous results directory, if exists.', default=False)
    parser.add_argument('--save_noise_pyramid', action='store_true',
                        help='for each sample, save its noise pyramid.',      default=False)
    parser.add_argument('--num_samples',        type=int,                     default=50,
                        help='Number of samples to generate at each run of random_samples.py')

    opt = parser.parse_args()
    opt = functions.post_config(opt)

    random_samples(opt)

