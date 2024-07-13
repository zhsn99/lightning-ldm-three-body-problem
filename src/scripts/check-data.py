import sys
sys.path.append('/home/zhsn/lightning-ldm/src')  # Add the directory containing the 'datasets' module
import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
from utils import *
import pudb
import pandas as pd
from datasets.Eye_contact_dataset import Eye_contact_dataset
from os.path import join as pjoin
import argparse
from omegaconf import OmegaConf
from utils import instantiate_from_config
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Eye_contact_dataset_loader(data.Dataset):
	def __init__(self, split):
		super().__init__()
		print(os.getcwd())
		self.dataset_root = f'../dataset/eye_contact_dataset_npy/{split}'
		files = os.listdir(self.dataset_root)
		files.sort()
		self.files = files
		self.seq_size = 30
        # set stuff here
		self.epoch = 0

	def init_dataset(self):
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
        # probably change this
		#/home/zhsn/projects/lightning-ldm/src/datasets
		#/home/zhsn/datasets/three_body_problem_data_npy files
		return len(self.files)

	def _set_seeds(self,idx):
		'''
		sets seeds based on epoch and item idx
		should be agnostic to worker number, rank within distributed
		'''  
		seed_gen = np.random.SeedSequence([self.epoch,idx])
		int_seed = int(seed_gen.generate_state(1))
		torch.manual_seed(int_seed)
		np.random.seed(int_seed)

	def __getitem__(self,idx):
		self._set_seeds(idx)
		#load npy file, 3 bodies are stacked on the same dimension
		seq = np.load(pjoin(self.dataset_root,self.files[idx]), allow_pickle = True).item() 
		seq = {key: value for key, value in seq.items() if 'win' not in key and 'RB' not in key}
		#convert each seq[key] values to a list then to an array
		seq = {key: np.array(list(value.values())) for key, value in seq.items()}
		
		len_seq = len(seq['frame'])

		gaze_direction_L = []
		gaze_direction_R = []
		#calculate gaze direction from gaze origin and gaze 3D
		# for row in range(len(seq['frame'])):
		# 	gaze_origin_left = [seq['T2.leftGazeOrigin.X'][row], seq['T2.leftGazeOrigin.Y'][row], seq['T2.leftGazeOrigin.Z'][row]]
		# 	gaze_origin_right = [seq['T2.rightGazeOrigin.X'][row], seq['T2.rightGazeOrigin.Y'][row], seq['T2.rightGazeOrigin.Z'][row]]
		# 	gaze_3d = [seq['T2.gaze3D.X'][row], seq['T2.gaze3D.Y'][row], seq['T2.gaze3D.Z'][row]]
		# 	gaze_direction_L_tmp = [gaze_3d[i] - gaze_origin_left[i] for i in range(3)]
		# 	gaze_direction_R_tmp = [gaze_3d[i] - gaze_origin_right[i] for i in range(3)]
        #     #make sure that the gaze direction is normalized
		# 	gaze_direction_L_tmp = gaze_direction_L_tmp / np.linalg.norm(gaze_direction_L_tmp)
		# 	gaze_direction_R_tmp = gaze_direction_R_tmp / np.linalg.norm(gaze_direction_R_tmp)
		# 	gaze_direction_L.append(gaze_direction_L_tmp)
		# 	gaze_direction_R.append(gaze_direction_R_tmp)

		# gaze_direction_L_x = np.array([gaze_direction_L[i][0] for i in range(len(gaze_direction_L))]) 
		# gaze_direction_L_y = np.array([gaze_direction_L[i][1] for i in range(len(gaze_direction_L))]) 
		# gaze_direction_L_z = np.array([gaze_direction_L[i][2] for i in range(len(gaze_direction_L))]) 
		# gaze_direction_R_x = np.array([gaze_direction_R[i][0] for i in range(len(gaze_direction_R))])
		# gaze_direction_R_y = np.array([gaze_direction_R[i][1] for i in range(len(gaze_direction_R))]) 
		# gaze_direction_R_z = np.array([gaze_direction_R[i][2] for i in range(len(gaze_direction_R))]) 
  
		head_rotation_x = seq['T2.RX']
		head_rotation_y = seq['T2.RY']
		head_rotation_z = seq['T2.RZ']
		head_rotation_w = seq['T2.RW']
        #find the first derivative of head rotation matrix
             
        
		tmp = {
            'index': self.files[idx],
			# 'frame': seq['frame'],
			# 'gaze_origin_left_x': seq['T2.leftGazeOrigin.X'] / 30,
			# 'gaze_origin_left_y': seq['T2.leftGazeOrigin.Y'] / 30,
			# 'gaze_origin_left_z': seq['T2.leftGazeOrigin.Z'] / 30,
			# 'gaze_origin_right_x': seq['T2.rightGazeOrigin.X'] / 30,
			# 'gaze_origin_right_y': seq['T2.rightGazeOrigin.Y'] / 30,
			# 'gaze_origin_right_z': seq['T2.rightGazeOrigin.Z'] / 30,
			# 'gaze_direction_left_x': gaze_direction_L_x,
			# 'gaze_direction_left_y': gaze_direction_L_y,
			# 'gaze_direction_left_z': gaze_direction_L_z,
			# 'gaze_direction_right_x': gaze_direction_R_x,
			# 'gaze_direction_right_y': gaze_direction_R_y,
			# 'gaze_direction_right_z': gaze_direction_R_z,
            # 'head_rotation_x': head_rotation_x,
            # 'head_rotation_y': head_rotation_y,
            # 'head_rotation_z': head_rotation_z,
            # 'head_rotation_w': head_rotation_w,
            'head_translation_x': seq['T2.TX'] / 1100,
            'head_translation_y': seq['T2.TY'] / 1100,
            'head_translation_z': seq['T2.TZ'] / 1100,
			# 'gaze_3d_x': seq['T2.gaze3D.X'],
			# 'gaze_3d_y': seq['T2.gaze3D.Y'],
			# 'gaze_3d_z': seq['T2.gaze3D.Z']
		}
		return tmp


def plot_rotation_values(dataset, data_dir):
    os.makedirs(f'{data_dir}/head_rotation', exist_ok=True)
    
    for i in range(len(dataset)):
        print(f'Processing {i}th data')

        # Extract quaternion data
        quaternions = np.array([
            dataset[i]['head_rotation_x'], 
            dataset[i]['head_rotation_y'], 
            dataset[i]['head_rotation_z'], 
            dataset[i]['head_rotation_w']
        ]).T

        print(f"Length of dataset {i}: {quaternions.shape[0]}")
        
        # Convert quaternions to rotation matrices using vectorized operations
        rotations = R.from_quat(quaternions)
        rot_mats = rotations.as_matrix()
        
        # Compute the difference between consecutive rotation matrices using vectorized operations
        inv_rot_mats = np.linalg.inv(rot_mats[:-1])
        relative_rotations = np.einsum('...ij,...jk->...ik', rot_mats[1:], inv_rot_mats)
        
        print(f"Length of rot_mats: {rot_mats.shape[0]}")
        print(f"Length of relative_rotations: {relative_rotations.shape[0]}")

        # Extract the rotation angles from the relative rotation matrices
        relative_rotations = R.from_matrix(relative_rotations)
        angles = relative_rotations.magnitude()
        
        print(f"Length of angles: {angles.shape[0]}")

        # Plot rotation angles over time
        time_frames_diff = np.arange(len(angles))
        print(f"Length of time_frames_diff: {time_frames_diff.shape[0]}")

        plt.figure(figsize=(12, 6))
        plt.plot(time_frames_diff, angles, label='Rotation Angle')
        plt.xlabel('Frame')
        plt.ylabel('Rotation Angle (radians)')
        plt.title('Relative Rotation Angles Over Time')
        plt.legend()
        plt.text(0, 0, dataset[i]['index'])

        plt.tight_layout()
        plt.savefig(f'{data_dir}/head_rotation/relative_rotation_angles_{i}.png')
        plt.close()  # Close the plot to free memory


def main():
    # make this script agnostic to cwd
    
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    src_path = os.path.abspath(os.path.join(script_path,'..'))
    sys.path.append(src_path)
    os.chdir(src_path)
    sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning
    argParser = argparse.ArgumentParser(description='')
    cli_args = argParser.parse_args()
    # load config
    core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
    base_config = OmegaConf.load(f'configs/simple_sequence_ddpm.yaml')
    config = OmegaConf.merge(core_config,base_config)
    data_dir = config.instance_data_dir
    #load data set

    dataset = Eye_contact_dataset_loader('train')
    #plot_rotation_values(dataset, data_dir)
    print(len(dataset))
    #plot data distribution for each dataset 
    
    for i in range(len(dataset)):
        print(f'Processing {i}th data')
        data = dataset[i]
        for key in data.keys():
            #number nans
            #if key == 'index'
            if key == 'index':
                continue
            os.makedirs(f'{data_dir}/T2/{key}/data_distribution',exist_ok=True)
            nn = np.sum(np.isnan(data[key]))
            plt.hist(data[key])
            index_file = dataset[i]['index']
            plt.title(f'{key} distribution for {i} data, {index_file}')
            plt.text(0, 0, f'nan: {nn}, percentage: {nn/len(data[key])}%')
            plt.xlabel(f'{key}')
            plt.ylabel('Frequency')
            plt.savefig(f'{data_dir}/T2/{key}/data_distribution/{key}_{i}.png')
            plt.close()
        print(f'Finished processing {i}th data')
    
    #plot head rotations, x,y,z,w in 4 subplots

        
        
        
     

main()