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
class Test_dataset(data.Dataset):
	def __init__(self, split):
		super().__init__()

		self.dataset_root = f'../dataset/synthetic_head_rotation/{split}'
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
		seq = np.load(pjoin(self.dataset_root,self.files[idx]), allow_pickle = True).item() #why!!!
		len_seq = len(seq['T1.RX'])
  
		crop_size = 480 
		start = np.random.randint(0,len_seq-crop_size)
		end = start + crop_size
		crop_seq = {key: value[start:end] for key, value in seq.items()}

		
		#check if there is any nan in crop_seq[keys] make it zero
		for key in crop_seq.keys():
			crop_seq[key] = np.nan_to_num(crop_seq[key])
		tmp = {
    		'head_rotation_x': crop_seq['T1.RX'],
			'head_rotation_y': crop_seq['T1.RY'],
			'head_rotation_z': crop_seq['T1.RZ'],
			'head_rotation_w': crop_seq['T1.RW'],
		}
		#time, 1(one body), gaze_origin, gaze_direction, gaze_3d
		#np.stack all of them to have a numpy array
		padding_zeros = np.zeros_like(tmp['head_rotation_x'])

		tmp = np.stack([tmp['head_rotation_x'], tmp['head_rotation_y'], tmp['head_rotation_z'],tmp['head_rotation_w'], padding_zeros, padding_zeros, padding_zeros, padding_zeros,padding_zeros,padding_zeros, padding_zeros, padding_zeros, padding_zeros, padding_zeros,padding_zeros,padding_zeros],axis = 1)
		bodies = tmp[None,...].astype(np.float32) #adding new dimension 
		#now bodies is agent, time, features
		#I want it to be agent, features, time
		bodies = bodies.transpose(0,2,1)
  
		out_dict = {
			'bodies':bodies
		}
		return out_dict

