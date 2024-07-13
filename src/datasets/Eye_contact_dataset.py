import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
import os
from utils import *
import pudb
import pandas as pd
# directory = '../../dataset/eye_contact_dataset/train'

# files = os.listdir(directory)

# for file in files:
#     print(file)
#     if '.csv' not in file:
#     	continue
#     seq = pd.read_csv(os.path.join(directory, file), encoding = 'utf-8')	
#     seq_dict = seq.to_dict()
# 	#print first key's values
#     print((seq_dict['frame']))
#     break
class Eye_contact_dataset(data.Dataset):
	def __init__(self, split):
		super().__init__()

		self.dataset_root = f'../dataset/eye_contact_dataset_npy/{split}_nparray'
		files = os.listdir(self.dataset_root)
		files.sort()
		self.files = files
		self.seq_size = 30
        # set stuff here
		self.epoch = 0
		self.dataset_repeats = 256
		self.npy_cache = {}

	def init_dataset(self):
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
        # probably change this
		#/home/zhsn/projects/lightning-ldm/src/datasets
		#/home/zhsn/datasets/three_body_problem_data_npy files
		return len(self.files) * self.dataset_repeats

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
		actual_idx = idx % len(self.files)
		self._set_seeds(actual_idx)
		#load npy file, 3 bodies are stacked on the same dimension
		if actual_idx in self.npy_cache.keys():
			seq = self.npy_cache[actual_idx]
		else:
			seq = np.load(pjoin(self.dataset_root,self.files[actual_idx]), allow_pickle = True).item() #why!!!
			self.npy_cache[actual_idx] = seq
		seq = {key: value for key, value in seq.items() if 'T2' not in key and 'win' not in key and 'RB' not in key}
		#convert each seq[key] values to a list then to an array
		# seq = {key: np.array(list(value.values())) for key, value in seq.items()}
		
		len_seq = len(seq['T1.RX'])
  
		crop_size = 960 # amount to skip at the native sample rate
		crop_skip = 1 # this value will also divide the final sample rate
		start = np.random.randint(0,len_seq-crop_size)
		end = start + crop_size
		crop_seq = {key: value[start:end:crop_skip] for key, value in seq.items()}

		
		#check if there is any nan in crop_seq[keys] make it zero
		for key in crop_seq.keys():
			crop_seq[key] = np.nan_to_num(crop_seq[key])
   
		gaze_direction_L = []
		gaze_direction_R = []
		#calculate gaze direction from gaze origin and gaze 3D
		for row in range(len(crop_seq['T1.RX'])):
			gaze_origin_left = [crop_seq['T1.leftGazeOrigin.XW'][row], crop_seq['T1.leftGazeOrigin.YW'][row], crop_seq['T1.leftGazeOrigin.ZW'][row]]
			gaze_origin_right = [crop_seq['T1.rightGazeOrigin.XW'][row], crop_seq['T1.rightGazeOrigin.YW'][row], crop_seq['T1.rightGazeOrigin.ZW'][row]]
			gaze_3d = [crop_seq['T1.gaze3D.XW'][row], crop_seq['T1.gaze3D.YW'][row], crop_seq['T1.gaze3D.ZW'][row]]
			gaze_direction_L_tmp = np.array([gaze_3d[i] - gaze_origin_left[i] for i in range(3)])
			gaze_direction_R_tmp = np.array([gaze_3d[i] - gaze_origin_right[i] for i in range(3)])
			gaze_direction_L_tmp = gaze_direction_L_tmp/np.linalg.norm(gaze_direction_L_tmp)
			gaze_direction_R_tmp = gaze_direction_R_tmp/np.linalg.norm(gaze_direction_R_tmp)
			if np.isnan(gaze_direction_L_tmp).any():
				gaze_direction_L_tmp = np.array([0,0,0])
			if np.isnan(gaze_direction_R_tmp).any():
				gaze_direction_R_tmp = np.array([0,0,0])
			gaze_direction_L.append(gaze_direction_L_tmp)
			gaze_direction_R.append(gaze_direction_R_tmp)

		gaze_direction_L_x = np.array([gaze_direction_L[i][0] for i in range(len(gaze_direction_L))])
		gaze_direction_L_y = np.array([gaze_direction_L[i][1] for i in range(len(gaze_direction_L))])
		gaze_direction_L_z = np.array([gaze_direction_L[i][2] for i in range(len(gaze_direction_L))])
		gaze_direction_R_x = np.array([gaze_direction_R[i][0] for i in range(len(gaze_direction_R))])
		gaze_direction_R_y = np.array([gaze_direction_R[i][1] for i in range(len(gaze_direction_R))])
		gaze_direction_R_z = np.array([gaze_direction_R[i][2] for i in range(len(gaze_direction_R))])
		tmp = {
			# 'frame': crop_seq['frame'],
			# 'gaze_origin_left_x': crop_seq['T1.leftGazeOrigin.X']/ 30,
			# 'gaze_origin_left_y': crop_seq['T1.leftGazeOrigin.Y']/ 30,
			# 'gaze_origin_left_z': crop_seq['T1.leftGazeOrigin.Z']/ 30,
			# 'gaze_origin_right_x': crop_seq['T1.rightGazeOrigin.X']/ 30,
			# 'gaze_origin_right_y': crop_seq['T1.rightGazeOrigin.Y']/ 30,
			# 'gaze_origin_right_z': crop_seq['T1.rightGazeOrigin.Z']/ 30,
			'gaze_direction_left_x': gaze_direction_L_x,
			'gaze_direction_left_y': gaze_direction_L_y,
			'gaze_direction_left_z': gaze_direction_L_z,
			'gaze_direction_right_x': gaze_direction_R_x,
			'gaze_direction_right_y': gaze_direction_R_y,
			'gaze_direction_right_z': gaze_direction_R_z,
    		'head_rotation_x': crop_seq['T1.RX'],
			'head_rotation_y': crop_seq['T1.RY'],
			'head_rotation_z': crop_seq['T1.RZ'],
			'head_rotation_w': crop_seq['T1.RW'],
			# 'head_translation_x': crop_seq['T1.TX'] / 1100,
			# 'head_translation_y': crop_seq['T1.TY'] / 1100,
			# 'head_translation_z': crop_seq['T1.TZ'] / 1100,
			# 'gaze_3d_x': crop_seq['T1.gaze3D.XW'],
			# 'gaze_3d_y': crop_seq['T1.gaze3D.YW'],
			# 'gaze_3d_z': crop_seq['T1.gaze3D.ZW']
		}
		#time, 1(one body), gaze_origin, gaze_direction, gaze_3d
		#np.stack all of them to have a numpy array
		padding_zeros = np.zeros_like(tmp['head_rotation_x'])

		tmp = np.stack([
			tmp['head_rotation_x'],
			tmp['head_rotation_y'],
			tmp['head_rotation_z'],
			tmp['head_rotation_w'],
			tmp['gaze_direction_left_x'],
			tmp['gaze_direction_left_y'],
			tmp['gaze_direction_left_z'],
			tmp['gaze_direction_right_x'],
			tmp['gaze_direction_right_y'],
			tmp['gaze_direction_right_z'],
			padding_zeros,
			padding_zeros,
			padding_zeros,
			padding_zeros,
			padding_zeros,
			padding_zeros
		],axis = 1)
		bodies = tmp[None,...].astype(np.float32) #adding new dimension 
		#now bodies is agent, time, features
		#I want it to be agent, features, time
		bodies = bodies.transpose(0,2,1)
  
		out_dict = {
			'bodies':bodies
		}
		return out_dict

