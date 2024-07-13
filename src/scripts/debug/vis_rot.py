
import os
import torch
import sys
sys.path.append('/home/jjyu/projects/quat-diff/src')  # Add the directory containing the 'datasets' module

# from datasets.Eye_contact_dataset import Eye_contact_dataset

import sys
from os.path import join as pjoin
import argparse
import numpy as np
from omegaconf import OmegaConf
from utils import instantiate_from_config
import scipy
import pudb
from PIL import Image, ImageDraw
import matplotlib

def vis_rot_coloured(quat):
    '''
    takes in Tensor of shape (Q,T)
    Q: quaterion dims
    T: rotations over time
    '''
    im_size = 256
    vec_len = 64
    quat = np.array(quat)
    T = quat.shape[1]
    base_vector = np.array([[0,0,1]]).T
    base_vector = np.array([[0,1,0]]).T

    # set up image
    im = Image.new('RGB',(im_size*3,im_size))
    draw = ImageDraw.Draw(im)
    draw.line((im_size, 0, im_size, im_size), fill='white')
    draw.line((im_size*2, 0, im_size*2, im_size), fill='white')
    draw.text((5,5),'XY','white')
    draw.text((im_size+5,0+5),'XZ','white')
    draw.text((im_size*2+5,0+5),'YZ','white')
    draw.line((5, im_size-5, 20, im_size-5), fill='red')
    draw.line((5, im_size-5, 5, im_size-20), fill='green')
    draw.line((im_size+5, im_size-5, im_size+20, im_size-5), fill='red')
    draw.line((im_size+5, im_size-5, im_size+5, im_size-20), fill='blue')
    draw.line((im_size*2+5, im_size-5, im_size*2+20, im_size-5), fill='green')
    draw.line((im_size*2+5, im_size-5, im_size*2+5, im_size-20), fill='blue')
    for t in range(T):
        norm_t = t/(T-1)
        cur_color = matplotlib.colormaps['brg'](norm_t)
        cur_color = tuple([int(c*255) for c in cur_color])

        # calculate rotated vectors
        q = quat[:,t]
        rot = scipy.spatial.transform.Rotation.from_quat(q)
        rotated = rot.as_matrix() @ base_vector
        rotated = (rotated*vec_len).flat

        # draw views
        xy_center = np.array([im_size//2,im_size//2])
        xy_endpoint = np.array([rotated[0],-rotated[1]]) + xy_center
        line_def = list(xy_center) + list(xy_endpoint)
        draw.line(line_def, fill=cur_color)

        xz_center = np.array([im_size + im_size//2,im_size//2])
        xz_endpoint = np.array([rotated[0],-rotated[2]]) + xz_center
        line_def = list(xz_center) + list(xz_endpoint)
        draw.line(line_def, fill=cur_color)

        yz_center = np.array([im_size*2 + im_size//2,im_size//2])
        yz_endpoint = np.array([rotated[1],-rotated[2]]) + yz_center
        line_def = list(yz_center) + list(yz_endpoint)
        draw.line(line_def, fill=cur_color)

    # im.save(f'coloured_rot.png')
    return im


def vis_rot_frames(quat):
    '''
    takes in Tensor of shape (Q,T)
    Q: quaterion dims
    T: rotations over time
    '''
    im_size = 256
    vec_len = 64
    quat = np.array(quat)
    T = quat.shape[1]
    base_vector = np.array([[0,0,1]]).T
    for t in range(T):
        # set up image
        im = Image.new('RGB',(im_size*3,im_size))
        draw = ImageDraw.Draw(im)
        draw.line((im_size, 0, im_size, im_size), fill='white')
        draw.line((im_size*2, 0, im_size*2, im_size), fill='white')
        draw.text((5,5),'XY','white')
        draw.text((im_size+5,0+5),'XZ','white')
        draw.text((im_size*2+5,0+5),'YZ','white')
        draw.line((5, im_size-5, 20, im_size-5), fill='red')
        draw.line((5, im_size-5, 5, im_size-20), fill='green')
        draw.line((im_size+5, im_size-5, im_size+20, im_size-5), fill='red')
        draw.line((im_size+5, im_size-5, im_size+5, im_size-20), fill='blue')
        draw.line((im_size*2+5, im_size-5, im_size*2+20, im_size-5), fill='green')
        draw.line((im_size*2+5, im_size-5, im_size*2+5, im_size-20), fill='blue')

        # calculate rotated vectors
        q = quat[:,t]
        rot = scipy.spatial.transform.Rotation.from_quat(q)
        rotated = rot.as_matrix() @ base_vector
        rotated = (rotated*vec_len).flat

        # draw views
        xy_center = np.array([im_size//2,im_size//2])
        xy_endpoint = np.array([rotated[0],-rotated[1]]) + xy_center
        line_def = list(xy_center) + list(xy_endpoint)
        draw.line(line_def, fill='white')

        xz_center = np.array([im_size + im_size//2,im_size//2])
        xz_endpoint = np.array([rotated[0],-rotated[2]]) + xz_center
        line_def = list(xz_center) + list(xz_endpoint)
        draw.line(line_def, fill='white')

        yz_center = np.array([im_size*2 + im_size//2,im_size//2])
        yz_endpoint = np.array([rotated[1],-rotated[2]]) + yz_center
        line_def = list(yz_center) + list(yz_endpoint)
        draw.line(line_def, fill='white')

        im.save(f'{t:04d}.png')

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
    src_path = os.path.abspath(os.path.join(script_path,'../..'))
    sys.path.append(src_path)
    os.chdir(src_path)
    sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning

    argParser = argparse.ArgumentParser(description='')
    cli_args = argParser.parse_args()

    from datasets.Rot_dataset import Rot_dataset
    from datasets.Eye_contact_dataset import Eye_contact_dataset

    # test the dataset here
    dataset = Eye_contact_dataset('train')
    # dataset = Rot_dataset('train')
    #get the first example
    for i in range(16):
        example = dataset[i]
        rots = example['bodies'][0,:4,:]
        # vis_rot_frames(rots)
        vis_rot_coloured(rots).save(f'z_{i:02d}.png')

    example = dataset[8]
    rots = example['bodies'][0,:4,:]
    print(rots.shape)
    vis_rot_frames(rots)




if __name__ == '__main__':
    main()
