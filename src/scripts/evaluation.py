import os
import sys
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd 
from scipy.stats import gaussian_kde

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# make this script agnostic to cwd
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)
sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning


# load config
core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
base_config = OmegaConf.load(f'configs/real_eye_head_rot_ddpm.yaml')
config = OmegaConf.merge(core_config,base_config)
data_dir = config.instance_data_dir
from utils import instantiate_from_config
import numpy as np


def angle_difference(a, b):
    angle_in_radians = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def gaze_direction_check(file, data, counter):
    #1,2,3,4 are head rotations
    #5,6,7 are left gaze direction
    #8,9,10 are right gaze direction
    gaze_direction_left = data[:,5:8]
    gaze_direction_right = data[:,8:11]
    gaze_direction_diff_list =[ ]
    for i in range(len(gaze_direction_left)):
        gaze_direction_left[i] = gaze_direction_left[i]/np.linalg.norm(gaze_direction_left[i])
        gaze_direction_right[i] = gaze_direction_right[i]/np.linalg.norm(gaze_direction_right[i])
        gaze_direction_diff = angle_difference(gaze_direction_left[i],gaze_direction_right[i])
        gaze_direction_diff_list.append(gaze_direction_diff)
        plt.plot(gaze_direction_diff_list)
        plt.ylim(-180, 180)
        plt.xlabel('Time')
        plt.ylabel('Gaze Direction Diffrence(degrees)')
        # plt.title(f'GDD {file}, AVG: {avg_gaze_direction_diff}')
        plt.savefig(f'{data_dir}/evaluation/sample_{counter}.png')
        plt.close()
        plt.clf()
    return gaze_direction_diff_list

# Plot histograms
def plot_histograms(rotations, labels, counter, evaluation_dir):
    print("plotting histograms...")
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, label in enumerate(labels):
        axs[i // 2, i % 2].hist(rotations[:, i], bins=30, alpha=0.7, label=label, color='blue')
        axs[i // 2, i % 2].set_title(f'Histogram of {label}')
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig(f'sample_histogram_{counter}.png')
    plt.clf()

def plot_kde(rotations, labels, counter, evaluation_dir):
    print("plotting KDE plots...")
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, label in enumerate(labels):
        kde = gaussian_kde(rotations[:, i])
        x = np.linspace(min(rotations[:, i]), max(rotations[:, i]), 1000)
        axs[i // 2, i % 2].plot(x, kde(x), label=label, color='red')
        axs[i // 2, i % 2].set_title(f'KDE of {label}')
        axs[i // 2, i % 2].legend()
    plt.tight_layout()
    plt.savefig(f'sample_kde_{counter}.png')
    plt.clf()
    
def head_rotation_check(data, counter, evaluation_dir):
    
    # Extract quaternion values (assuming they are in columns 1 to 4)
    head_rotations = data[:, 1:5]
    labels = ['x', 'y', 'z', 'w']

    # Call the plotting functions
    print(f"Plotting head rotation histograms and KDE plots for {counter}...")
    plot_histograms(head_rotations, labels, counter, evaluation_dir)
    plot_kde(head_rotations, labels, counter, evaluation_dir)
    

# Main execution function
def main():
    data_dir = config.instance_data_dir
    os.makedirs(f'{data_dir}/evaluation', exist_ok=True)
    evaluation_dir = os.path.join(data_dir,'evaluation')
    data_files = os.path.join(data_dir,'results')
    csv_files = os.listdir(data_files)
    #check files that end with .csv and dont contain unity in their name
    csv_files = [file for file in csv_files if file.endswith('.csv') and 'unity' not in file]
    counter = 0
    for counter, file in enumerate(csv_files, 1):
        df = pd.read_csv(os.path.join(data_files, file))
        img_np = df.to_numpy()
        print(img_np.shape)
        # head_rotation_check(img_np, counter, evaluation_dir)
        # plot gaze direction diffrence over time
        gaze_direction_diff_list = gaze_direction_check(file, img_np, counter)
        avg_gaze_direction_diff = np.mean(gaze_direction_diff_list)
        
        counter += 1
   
        

if __name__ == '__main__':
    main()
