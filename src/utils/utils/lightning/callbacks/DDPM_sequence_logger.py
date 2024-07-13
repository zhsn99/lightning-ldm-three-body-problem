import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import torch
import torchvision
import numpy as np
import os
from PIL import Image, ImageDraw
import wandb
import pudb
import matplotlib
import scipy

def vis_dir_coloured(dirs):
    '''
    takes in Tensor of shape (3,T)
    3: 3D vector elements
    T: rotations over time
    '''
    im_size = 256
    vec_len = 64
    dirs = np.array(dirs)
    T = dirs.shape[1]

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

        # extract direction
        direction = dirs[:,t]*vec_len

        # draw views
        xy_center = np.array([im_size//2,im_size//2])
        xy_endpoint = np.array([direction[0],-direction[1]]) + xy_center
        line_def = list(xy_center) + list(xy_endpoint)
        draw.line(line_def, fill=cur_color)

        xz_center = np.array([im_size + im_size//2,im_size//2])
        xz_endpoint = np.array([direction[0],-direction[2]]) + xz_center
        line_def = list(xz_center) + list(xz_endpoint)
        draw.line(line_def, fill=cur_color)

        yz_center = np.array([im_size*2 + im_size//2,im_size//2])
        yz_endpoint = np.array([direction[1],-direction[2]]) + yz_center
        line_def = list(yz_center) + list(yz_endpoint)
        draw.line(line_def, fill=cur_color)

    return im

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

    return im

class DDPM_sequence_logger(Callback):
    def __init__(self, batch_frequency, increase_log_steps):
        super().__init__()
        self.batch_freq = batch_frequency
        
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
            
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if self.check_frequency(batch_idx):
            is_train = pl_module.training # remember train state
            if is_train: pl_module.eval()

            with torch.no_grad():
                sample_shape = [1]+list(batch['bodies'].shape[1:])
                sampled_data = pl_module.p_sample_loop(sample_shape)
                
                quaternion_norm = torch.norm(sampled_data[0,0,0:4,0]).item()
                pl_module.log(f"{split}_quaternion_norm_t4", quaternion_norm)

                left_gaze_dir = torch.Tensor(sampled_data[0,0,4:7,:]).cpu().detach().numpy()
                right_gaze_dir = torch.Tensor(sampled_data[0,0,7:10,:]).cpu().detach().numpy()

                rot_vis = vis_rot_coloured(sampled_data[0,0,:4,:].cpu().detach().numpy())
                left_gaze_vis = vis_dir_coloured(left_gaze_dir)
                right_gaze_vis = vis_dir_coloured(right_gaze_dir)
                
                # global step is redefined as of lightning 1.6
                actual_global_step = pl_module.trainer.fit_loop.epoch_loop.total_batch_idx
                log_dict = {
                    f"{split}_quaternion_norm_t4": quaternion_norm,
                    'rotation_vis': wandb.Image(rot_vis),
                    'left_gaze_vis': wandb.Image(left_gaze_vis),
                    'right_gaze_vis': wandb.Image(right_gaze_vis),
                    'trainer/global_step': actual_global_step
                }
                # pl_module.log_dict(log_dict)
                pl_module.logger.experiment.log(log_dict)

            if is_train: pl_module.train() # restore train state

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")
