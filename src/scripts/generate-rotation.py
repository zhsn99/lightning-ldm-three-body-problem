from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# Function to generate random quaternion with small deviation from the previous quaternion
# def generate_smooth_quaternion(previous_quaternion, step_size=0.01):
#     random_vector = np.random.normal(0, step_size, 3)  # small random changes in x, y, z
#     random_quaternion = R.from_rotvec(random_vector).as_quat()
#     new_quaternion = R.from_quat(previous_quaternion) * R.from_quat(random_quaternion)
#     #print norm of new_quaternion
#     return new_quaternion.as_quat()

# def generate(idx):
#     num_frames = np.random.randint(5000, 10000)
#     step_size = 0.01
#     start_quaternion = R.random().as_quat()
#     quaternions = np.zeros((num_frames, 4))
#     quaternions[0] = start_quaternion
#     for i in range(1, num_frames):
#         quaternions[i] = generate_smooth_quaternion(quaternions[i-1], step_size)

#     #save as a .npy file for later use

#     #create a dict T1.RX, T1.RY, T1.RZ, T1.RW

#     quaternion_dict = {
#         'T1.RX': quaternions[:, 0],
#         'T1.RY': quaternions[:, 1],
#         'T1.RZ': quaternions[:, 2],
#         'T1.RW': quaternions[:, 3],
#     }

#     np.save(f'../../dataset/synthetic_head_rotation/synthetic_head_rotation_{idx+1}.npy', quaternion_dict)

# for i in tqdm(range(1000)):
#     generate(i)
#load data from ../../dataset/synthetic_head_rotation/synthetic_head_rotation_{idx+1}.npy plot their norm of quartenion over time

def plot_quaternion_norm(data_dir):
    files = os.listdir(data_dir)
    for file in files:
        if not file.endswith('.npy'):
            continue
        seq = np.load(os.path.join(data_dir, file), allow_pickle=True).item()
        quaternions = np.array([seq['T1.RX'], seq['T1.RY'], seq['T1.RZ'], seq['T1.RW']]).T
        norm = np.linalg.norm(quaternions, axis=1)
        plt.plot(norm)
        plt.xlabel('Time')
        plt.ylabel('Norm of quaternion')
        plt.title('Norm of quaternion over time')
        plt.savefig(f'{data_dir}/{file}.png')
        plt.close()
data_dir = '../../dataset/synthetic_head_rotation/train'
plot_quaternion_norm(data_dir)