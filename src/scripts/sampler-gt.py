import os
import sys
from os.path import join as pjoin
import argparse
import pudb
import os
import pandas as pd 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# make this script agnostic to cwd
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)
sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning

import torch
import matplotlib.pyplot as plt
from utils import instantiate_from_config
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets.Eye_contact_dataset import Eye_contact_dataset
import random
import pudb
import numpy as np

# load config
core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
base_config = OmegaConf.load(f'configs/new_simple_sequence_ddpm.yaml')
config = OmegaConf.merge(core_config,base_config)
data_dir = config.instance_data_dir

dataset = Eye_contact_dataset('train')


def sample_from_model(model, device, num_samples=10):
    global dataset
    max_idx = len(dataset)
    examples = []
    for i in range(num_samples):
        idx = random.randint(0, max_idx)
        examples.append(torch.Tensor(dataset[idx]['bodies'].astype(np.float32)))
    return torch.stack(examples)
    # return model.p_sample_loop((num_samples, 3, 120, 2), return_intermediates=False)

def main():
    print("Sampling from ground truth data")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampled_images = sample_from_model(None, device, num_samples=10) 
    #make image to be Sampled images shape: torch.Size([1, 480, 16])
    sampled_images = sampled_images.permute(0,1,3,2)
    print(sampled_images.shape)
    #create a results directory in scripts/results
    os.makedirs(f'{data_dir}/results-gt',exist_ok=True)
    counter =0
    for img in sampled_images:
        print(img.shape)
        print(img[0].shape)
        img_cpu = img.cpu()  # move tensor to CPU
        img_np = img_cpu.numpy()  # convert tensor to numpy array
        img_np = img_np.squeeze(0)
        print(img_np.shape)
        df = pd.DataFrame(img_np)
        df.to_csv(f'{data_dir}/results-gt/sample_{counter}.csv')     
        counter+=1  
#evaluation ideas MSE, FID, Visualization, etc
#plot groundtruth and generated path
#plot the generated path --> done
#how to plot coresponding groundtruth? also for doing MSE and FID

def mean_squared_error():
    pass

def frechet_inception_distance():
    pass

def visualization():
    pass
if __name__ == '__main__':
    main()
