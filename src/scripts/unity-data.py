import os
import sys
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd 
import scipy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# make this script agnostic to cwd
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)
sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from utils import instantiate_from_config
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
# load config
core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
base_config = OmegaConf.load(f'configs/real_eye_head_rot_ddpm.yaml')
config = OmegaConf.merge(core_config,base_config)
data_dir = config.instance_data_dir



results_dir = os.path.join(data_dir,'results')
csv_files = os.listdir(results_dir)
# column_map = {
#     'Unnamed: 0': 'frame',
#     '0': 'T1.leftGazeOrigin.X',
#     '1': 'T1.leftGazeOrigin.Y',
#     '2': 'T1.leftGazeOrigin.Z',
#     '3': 'T1.rightGazeOrigin.X',
#     '4': 'T1.rightGazeOrigin.Y',
#     '5': 'T1.rightGazeOrigin.Z',
#     '6': 'T1.leftGazeDirection.X',
#     '7': 'T1.leftGazeDirection.Y',
#     '8': 'T1.leftGazeDirection.Z',
#     '9': 'T1.rightGazeDirection.X',
#     '10': 'T1.rightGazeDirection.Y',
#     '11': 'T1.rightGazeDirection.Z',
#     '12': 'T1.RX',
#     '13': 'T1.RY',
#     '14': 'T1.RZ',
#     '15': 'T1.RW',
#     '16': 'T1.TX',
#     '17': 'T1.TY',
#     '18': 'T1.TZ',
# }
column_map = {
    'Unnamed: 0': 'frame',
    '0': 'T1.RX',
    '1': 'T1.RY',
    '2': 'T1.RZ',
    '3': 'T1.RW',
    '4': 'T1.leftGazeDirection.XW',
    '5': 'T1.leftGazeDirection.YW',
    '6': 'T1.leftGazeDirection.ZW',
    '7': 'T1.rightGazeDirection.XW',
    '8': 'T1.rightGazeDirection.YW',
    '9': 'T1.rightGazeDirection.ZW',
    
}

def local2global(seq):
    # #add this new columns to seq 
    # seq['T1.leftGazeOrigin.XW'] = 0
    # seq['T1.leftGazeOrigin.YW'] = 0
    # seq['T1.leftGazeOrigin.ZW'] = 0
    # seq['T1.rightGazeOrigin.XW'] = 0
    # seq['T1.rightGazeOrigin.YW'] = 0
    # seq['T1.rightGazeOrigin.ZW'] = 0
    # seq['T1.leftGazeDirection.XW'] = 0
    # seq['T1.leftGazeDirection.YW'] = 0
    # seq['T1.leftGazeDirection.ZW'] = 0
    # seq['T1.rightGazeDirection.XW'] = 0
    # seq['T1.rightGazeDirection.YW'] = 0
    # seq['T1.rightGazeDirection.ZW'] = 0
    
    # seq['T1.TX'] = seq['T1.TX'] * 1100
    # seq['T1.TY'] = seq['T1.TY'] * 1100
    # seq['T1.TZ'] = seq['T1.TZ'] * 1100
    
    # seq['T1.leftGazeOrigin.X'] = seq['T1.leftGazeOrigin.X'] * 30
    # seq['T1.leftGazeOrigin.Y'] = seq['T1.leftGazeOrigin.Y'] * 30
    # seq['T1.leftGazeOrigin.Z'] = seq['T1.leftGazeOrigin.Z'] * 30
    
    for row in tqdm(range(len(seq['T1.RX']))):

        qx = seq['T1.RX'][row]
        qy = seq['T1.RY'][row]
        qz = seq['T1.RZ'][row]
        qw = seq['T1.RW'][row]

        # # Calculate the norm of the quaternion
        # norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)

        # # Normalize the quaternion
        # qx_normalized = qx / norm
        # qy_normalized = qy / norm
        # qz_normalized = qz / norm
        # qw_normalized = qw / norm
        
        seq['T1.RX'][row] = qx
        seq['T1.RY'][row] = qy
        seq['T1.RZ'][row] = qz
        seq['T1.RW'][row] = qw
        

        # # Create a rotation object from the normalized quaternion
        # rot_mat = R.from_quat([qx_normalized, qy_normalized, qz_normalized, qw_normalized]).as_matrix()
        # translation = np.array([seq['T1.TX'][row], seq['T1.TY'][row], seq['T1.TZ'][row]])
        
        # # Apply rotation to gaze directions
        # gaze_dir_vector = np.array([seq['T1.leftGazeDirection.X'][row], seq['T1.leftGazeDirection.Y'][row], seq['T1.leftGazeDirection.Z'][row]])
        # gaze_dir_vector = rot_mat @ gaze_dir_vector
        # seq['T1.leftGazeDirection.XW'][row] = gaze_dir_vector[0]
        # seq['T1.leftGazeDirection.YW'][row] = gaze_dir_vector[1]
        # seq['T1.leftGazeDirection.ZW'][row] = gaze_dir_vector[2]
        
        # gaze_dir_vector = np.array([seq['T1.rightGazeDirection.X'][row], seq['T1.rightGazeDirection.Y'][row], seq['T1.rightGazeDirection.Z'][row]])
        # gaze_dir_vector = rot_mat @ gaze_dir_vector
        # seq['T1.rightGazeDirection.XW'][row] = gaze_dir_vector[0]
        # seq['T1.rightGazeDirection.YW'][row] = gaze_dir_vector[1]
        # seq['T1.rightGazeDirection.ZW'][row] = gaze_dir_vector[2]
        
        # # Apply rotation to gaze origins
        # gaze_origin = np.array([seq['T1.leftGazeOrigin.X'][row], seq['T1.leftGazeOrigin.Y'][row], seq['T1.leftGazeOrigin.Z'][row]])
        # gaze_origin = rot_mat @ gaze_origin + translation
        # seq['T1.leftGazeOrigin.XW'][row] = gaze_origin[0]
        # seq['T1.leftGazeOrigin.YW'][row] = gaze_origin[1]
        # seq['T1.leftGazeOrigin.ZW'][row] = gaze_origin[2]
        
        # gaze_origin = np.array([seq['T1.rightGazeOrigin.X'][row], seq['T1.rightGazeOrigin.Y'][row], seq['T1.rightGazeOrigin.Z'][row]])
        # gaze_origin = rot_mat @ gaze_origin + translation
        # seq['T1.rightGazeOrigin.XW'][row] = gaze_origin[0]
        # seq['T1.rightGazeOrigin.YW'][row] = gaze_origin[1]
        # seq['T1.rightGazeOrigin.ZW'][row] = gaze_origin[2]
        
    return seq
        
        
        


def main():
    for file in csv_files:
        print(file)
        if '.csv' not in file:
            continue
        if '-unity' in file:
            continue
        seq = pd.read_csv(os.path.join(results_dir, file), encoding = 'utf-8')
        #drop last five columns
        print("this is the columns")
        print(seq.columns)
        seq = seq.rename(columns=column_map)
        seq = local2global(seq)
        #save the new seq to the new file add -unity to the name
        seq.to_csv(os.path.join(results_dir, file.replace('.csv', '-unity.csv')), index=False)
        

main()
    

