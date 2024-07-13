import os
import sys
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd 




def angle_difference(a, b):
    angle_in_radians = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def gaze_direction_check(data):
    gaze_direction_left = data[:,6:9]
    gaze_direction_right = data[:,9:12]
    gaze_direction_left = gaze_direction_left/np.linalg.norm(gaze_direction_left)
    gaze_direction_right = gaze_direction_right/np.linalg.norm(gaze_direction_right)
    gaze_direction_diff_list =[]
    for i in range(len(gaze_direction_left)):
        gaze_direction_diff = angle_difference(gaze_direction_left[i],gaze_direction_right[i])
        gaze_direction_diff_list.append(gaze_direction_diff)
    return gaze_direction_diff_list

# Sample from model
def sample_from_model(model, device, num_samples=10):
    return model.p_sample_loop((num_samples, 1, 16, 600), return_intermediates=False)

# Main execution function
def main():
    #python sampler.y xxx.yaml
    config_path = sys.argv[1]
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # make this script agnostic to cwd
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    src_path = os.path.abspath(os.path.join(script_path,'..'))
    sys.path.append(src_path)
    os.chdir(src_path)
    sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning


    # load config
    core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
    base_config = OmegaConf.load(f'configs/{config_path}')
    config = OmegaConf.merge(core_config,base_config)
    data_dir = config.instance_data_dir
    from utils import instantiate_from_config
    import numpy as np
    
    print("Sampling from model")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = f'{data_dir}/checkpoints/epoch=1070-step=00000000.ckpt'
    config = OmegaConf.merge(
        OmegaConf.load('configs/core/training_mandatory.yaml'),
        OmegaConf.load(f'configs/{config_path}')
    )
    model = instantiate_from_config(config.model)
    print("Loading model from", model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print("******Model loaded*******")
    sampled_images = sample_from_model(model, device, num_samples=10)
    print("Sampled images shape before:", sampled_images.shape)
    
    print("Sampled images shape before:", sampled_images.shape)
    #make image to be Sampled images shape: torch.Size([10, 1, 600, 16])
    sampled_images = sampled_images.permute(0, 1, 3, 2)
    print("Sampled images shape after:", sampled_images.shape)
    # Create a results directory
    data_dir = config.instance_data_dir
    os.makedirs(f'{data_dir}/results', exist_ok=True)

    for counter, img in enumerate(sampled_images, 1):
        print("Processing sample", counter)
        # print("Processing sample", counter)
        
        img_cpu = img.cpu()
        img_np = img_cpu.numpy()
        img_np = img_np.squeeze(0)

        df = pd.DataFrame(img_np)
        df.to_csv(f'{data_dir}/results/sample_{counter}.csv')
   
        

if __name__ == '__main__':
    main()
