import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
from utils import *

class Clevr_dataset(data.Dataset):
	def __init__(self,split):
		super().__init__()
		self.dataset_root = f'../dataset-data/data/{split}'
		self.image_root = pjoin(self.dataset_root,'images')
		self.transform_root = pjoin(self.dataset_root,'transforms')
		files = os.listdir(self.transform_root)
		files.sort()
		self.files = files
		self.im_size = 128
		self.epoch = 0

	def init_dataset():
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
		return len(self.files)

	def _set_seeds(self,idx):
		'''
		sets seeds based on epoch and item idx
		should be agnostic to worker number, rank within distributed
		'''
		seed = self.epoch*5000000 + idx
		torch.manual_seed(seed)
		np.random.seed(seed)

	def __getitem__(self,idx):
		self._set_seeds(idx)
		transform_path = pjoin(self.transform_root,self.files[idx])
		image_idx = int(self.files[idx][:-5])
		image_fns = [f'CLEVR_new_{image_idx:06d}-a.png',f'CLEVR_new_{image_idx:06d}-b.png']
		image_paths = [pjoin(self.image_root,x) for x in image_fns]
		
		im_choice = np.random.choice(2)

		# stacked images
		im = Image.open(image_paths[im_choice])
		im = (np.asarray(im) /255 - 0.5)[...,:3]
		im = im.transpose(2,0,1).astype(np.float32)

		out_dict = {
			'im': im,
		}
		return out_dict

