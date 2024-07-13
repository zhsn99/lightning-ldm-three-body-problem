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
import scipy

class Rot_dataset(data.Dataset):
	def __init__(self, split):
		super().__init__()

		self.size = 100000
		self.seq_size = 32
		self.epoch = 0

	def init_dataset(self):
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
		return self.size

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

		# generate random rotation matrix
		rand_rot = scipy.spatial.transform.Rotation.random()
		rotvec = rand_rot.as_rotvec()
		norm = np.sqrt((rotvec**2).sum())
		rotvec /= norm
		rotvec *= np.pi*2*1.02

		# generate a random 3d vector
		vec = np.random.rand(3)
		norm = np.sqrt((vec**2).sum())
		vec /= norm

		# repeatedly rotate the random vector using our rotation mat
		base_rot = scipy.spatial.transform.Rotation.from_rotvec(rotvec)
		rots = []
		for n in range(self.seq_size):
			cumulative_rot = base_rot
			for n2 in range(n):
				cumulative_rot *= base_rot
			# rots.append(cumulative_rot.as_matrix() @ vec)
			rots.append(cumulative_rot.as_quat())
	
		# data shape: agent, time, features
		# rots = np.pad(rots,((0,0),(0,1)))
		bodies = np.stack(rots,-1)[None,...].astype(np.float32)

		out_dict = {
			'bodies':bodies
		}
		return out_dict

