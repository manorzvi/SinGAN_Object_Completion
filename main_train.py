from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import shutil

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir',                      help='input image dir', default='Input/Images')
    parser.add_argument('--input_name',      required=True, help='input image name')
    parser.add_argument('--mode',                           help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)

    Gs       = []
    Zs       = []
    reals    = []
    NoiseAmp = []

    dir2save = functions.generate_dir2save(opt)
    print(dir2save)
    if (os.path.exists(dir2save)):
        if opt.delete_previous:
            shutil.rmtree(dir2save)
        else:
            print('trained model already exist')
            exit(1)

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)
    SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)

    exit(0)