
import os
import torch
import sys
sys.path.append('/home/zhsn/lightning-ldm/src')  # Add the directory containing the 'datasets' module

from datasets.Eye_contact_dataset import Eye_contact_dataset


import sys
from os.path import join as pjoin
import argparse
import numpy as np
from omegaconf import OmegaConf
from utils import instantiate_from_config
import pudb
import pandas as pd


def main():
    '''
    This script glues all the things we need for distributed training
     - loading configs
     - resuming from checkpoints
     - custom checkpointing behavior
     - logging with wandb
     - setting up the datamodule with extra stuff for mid epoch resume
     - setting up LR based on distributed env
     - printing config to stdout on startup
     - attaching custom callbacks
     - building and running the trainer
    '''

    # make this script agnostic to cwd
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    src_path = os.path.abspath(os.path.join(script_path,'..'))
    sys.path.append(src_path)
    os.chdir(src_path)
    sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning

    argParser = argparse.ArgumentParser(description='')
    cli_args = argParser.parse_args()


    # test the dataset here
    dataset = Eye_contact_dataset('train')
    #get the first example
    example = dataset[0]

    # pu.db
    
    # for i in range(480):
    #     qurtenion_values = example['bodies'][0][:4,i]
    #     #print norm of qurtenion
    #     print("ARRRRR")
    #     print(np.linalg.norm(qurtenion_values))
	# 	tmp = np.stack([
	# 		tmp['head_rotation_x'],
	# 		tmp['head_rotation_y'],
	# 		tmp['head_rotation_z'],
	# 		tmp['head_rotation_w'],
	# 		tmp['gaze_direction_left_x'],
	# 		tmp['gaze_direction_left_y'],
	# 		tmp['gaze_direction_left_z'],
	# 		tmp['gaze_direction_right_x'],
	# 		tmp['gaze_direction_right_y'],
	# 		tmp['gaze_direction_right_z'],
	# 		padding_zeros,
	# 		padding_zeros,
	# 		padding_zeros,
	# 		padding_zeros,
	# 		padding_zeros,
	# 		padding_zeros
	# 	],axis = 1)
	# 	bodies = tmp[None,...].astype(np.float32) #adding new dimension 
	# 	#now bodies is agent, time, features
	# 	#I want it to be agent, features, time
	# 	bodies = bodies.transpose(0,2,1)
  
	# 	out_dict = {
	# 		'bodies':bodies
	# 	}       
    print(len(dataset))
    for i in range(500):
       
        #df columns : 'T1.RX', 'T1.RY', 'T1.RZ', 'T1.RW', T1.leftGazeDirection.XW, T1.leftGazeDirection.YW, T1.leftGazeDirection.ZW, T1.rightGazeDirection.XW, T1.rightGazeDirection.YW, T1.rightGazeDirection.ZW
        
        example = dataset[i]
        example = example['bodies'][0].T
        #keep 0-9 and drop 10-15 columns
        example = example[:,0:10]
        example = example.astype(float)
        #add frame numbers to the first column 0 to len(example)
        #make sure frame is int
        example = np.insert(example, 0, np.arange(len(example)), axis=1)
        df = pd.DataFrame(data=example, columns=['frame','T1.RX', 'T1.RY', 'T1.RZ', 'T1.RW', 'T1.leftGazeDirection.XW', 'T1.leftGazeDirection.YW', 'T1.leftGazeDirection.ZW', 'T1.rightGazeDirection.XW', 'T1.rightGazeDirection.YW', 'T1.rightGazeDirection.ZW'])
        #print type of a number in df
        df['frame'] = df['frame'].astype(int)
        os.makedirs('../instance-data-real_eye_head_rot_ddpm/processed', exist_ok=True)
        print(df.dtypes)
        df.to_csv(f'../instance-data-real_eye_head_rot_ddpm/processed/df_{i}.csv', index=False)
    
        
            
    # example = dataset[0]
    # print(example['bodies'][0])
    # # input_formatter(example)

    # # dataset works now, build a model to test
    # # load config
    # config = OmegaConf.load(f'configs/test_simple_sequence_ddpm.yaml')
    # model_cfg = config.model
    
    # model = instantiate_from_config(model_cfg)

    # # push data into model to check shapes
    # t = torch.randint(0, 500, (1,), device='cpu').long()
    # x = torch.tensor(example['bodies']).unsqueeze(0).float()
    # a,b,c,d = x.shape
    # print(x.shape)
    # print(t.shape)
    # try:
    #     model_out = model.denoiser(x, t)
    #     print(model_out.shape)
    # except:
    #     pudb.pm()
    print('done!')





if __name__ == '__main__':
    main()
