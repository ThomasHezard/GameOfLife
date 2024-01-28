# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import random
from tqdm.auto import tqdm
import os, shutil, subprocess, math

# Parameters
grid_shape = [1080, 1920]
half_kernel_size = 128
kernel_size = half_kernel_size * 2 + 1
nb_steps = 100
time_step = 0.1
video_size_factor = 1
video_frame_rate = 5
tmp_dir = 'img'


# *********************************************************

def init_random_grid(grid):
  for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
      grid[i, j] = float(random.uniform(0.0, 1.0))

def init_grid(grid):
  init_random_grid(grid)

mu = 0.3
sigma = 0.03
def growth_mapping(value):
  return 2*np.exp(-(value-mu)**2/(2*(sigma**2))) - 1

alpha = 4.0
kernel = np.zeros([kernel_size, kernel_size])
for i in range(kernel_size):
  for j in range(kernel_size):
    # # exponential kernel -- NOT WORKING
    # r = math.sqrt((i / (kernel_size-1.0)) ** 2.0 + (j - (kernel_size-1.0)) ** 2.0)
    # if r == 0.0 or r == 1.0:
    #   kernel[i,j] = 0
    # else:
    #   kernel[i,j] = math.exp(alpha - alpha / (4.0*r*(1-r)))
    # ring kernel
    r = math.sqrt((i / (kernel_size-1.0)) ** 2.0 + (j - (kernel_size-1.0)) ** 2.0)
    if r >= 1.0/4.0 and r <= 3.0/4.0:
      kernel[i,j] = (1.0-math.cos(((r-1.0/4.0)*2.0)*math.pi))/2.0
plt.imsave("kernel.png", kernel)
#kernel = kernel / np.sum(kernel)
print(np.max(kernel))
print(np.min(kernel))

# def update_grid(grid):
#   conv_grid = convolve2d(grid, kernel, mode='same', boundary='wrap')
#   grid += growth_mapping(conv_grid)
#   grid[grid > 1.0] = 1.0
#   grid[grid < 0.0] = 0.0

# grid = np.zeros(grid_shape)
# init_grid(grid)

# shutil.rmtree(tmp_dir, ignore_errors=True)
# os.mkdir(tmp_dir)

# for step in tqdm(range(nb_steps)):
#   filename = f"img/im_{step:06d}.png"
#   plt.imsave(filename, grid)
#   update_grid(grid)  

# cmd = f"ffmpeg -y -framerate {video_frame_rate} -pattern_type glob -i '{tmp_dir}/*.png' -vf scale={grid_shape[0]*video_size_factor}:{grid_shape[1]*video_size_factor}:flags=neighbor -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -movflags +faststart out.mp4"
# res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
# shutil.rmtree(tmp_dir, ignore_errors=True)
