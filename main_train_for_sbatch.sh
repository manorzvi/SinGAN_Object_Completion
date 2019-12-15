#!/bin/bash

# Setup env
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cs236781-hw
echo "hello from $(python --version) in $(which python)"

python -c 'import torch; print(f"Am I playing on the Cuda? {torch.cuda.is_available()}")'
python -c 'print("This is the real Shit!")'
python -c 'print("We are harvesting the power of CS faculty for our needs! ya we are hackers!")'

python main_train.py --input_name crowded1.jpg --input_dir Input/People --min_size 50 --max_size 750
python main_train.py --input_name panda.jpg    --input_dir Input/People --min_size 50 --max_size 750
python main_train.py --input_name urban2.jpg   --input_dir Input/People --min_size 50 --max_size 750



