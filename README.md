SinGAN imp. 4 object completion in images using latent space arithmetics

for noise pyramid saving:
random_samples.py --input_name balloons.png --input_dir Input/Images --mode random_samples --gen_start_scale 0 --delete_previous --not_cuda --num_samples 10 --save_noise_pyramid

note: while traning is mandatory to being performed on GPU only, Inference can be done on either GPU or CPU.

To create virtual env (after git clone):
conda env create -f environment.yml
