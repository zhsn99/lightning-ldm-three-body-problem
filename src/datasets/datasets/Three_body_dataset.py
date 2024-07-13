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

class Three_body_dataset(data.Dataset):
	def __init__(self, split):
		super().__init__()
		self.dataset_root = f'..//..//..//datasets//three_body_problem_data_npy//{split}'
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
		seq = np.load(pjoin(self.dataset_root,self.files[idx]))
		all_len_seq = len(seq)
		body_one  = seq[0:all_len_seq//3]
		body_two  = seq[all_len_seq//3:2*all_len_seq//3]
		body_three  = seq[2*all_len_seq//3:]
		len_seq = len(body_one)
  

		# stack bodies onto new dim, sort frame idx, remove aux data from rows
		bodies = np.stack((body_one, body_two, body_three), axis=1)
		bodies = bodies[:,:,2:]
		bodies = bodies[::32,...] # subsample
		len_seq = bodies.shape[0]

		# random crop on time dim
		crop_size = 120 # 120
		if len_seq <= crop_size:
			print(len_seq, crop_size)
   
		start = np.random.randint(0,len_seq-crop_size)
		bodies = bodies[start:start+crop_size]

		# permute to body, time, x/y
		bodies = bodies.transpose(1,0,2).astype(np.float32)
		# bodies = np.pad(bodies, ((0,0),(0,0),(0,1)), 'constant', constant_values=0)

		# output format: time, body, x/y
		# a random 120 frames of data
		out_dict = {
			'bodies':bodies
		}
		return out_dict

