# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import random
import time
from tqdm.auto import tqdm
import os, shutil, subprocess

# Parameters
init_proba = 0.2
grid_shape = [1080, 1920]
nb_steps = 1000
time_step = 0.1
video_size_factor = 1
video_frame_rate = 5
tmp_dir = 'img'


# *********************************************************

def init_random_grid(grid, init_proba):
  for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
      grid[i, j] = float(random.uniform(0.0, 1.0) < init_proba)

def init_grid(grid):
  init_random_grid(grid, init_proba)

kernel = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])
def update_grid(grid):
  conv_grid = convolve2d(grid, kernel, mode='same', boundary='wrap')
  death_indexes = np.logical_and(grid == 1.0, np.logical_or(conv_grid < 2, conv_grid > 3))
  birth_indexes = np.logical_and(grid == 0.0, conv_grid == 3)
  grid[death_indexes] = 0.0
  grid[birth_indexes] = 1.0

grid = np.zeros(grid_shape)
init_grid(grid)

shutil.rmtree(tmp_dir, ignore_errors=True)
os.mkdir(tmp_dir)

for step in tqdm(range(nb_steps)):
  filename = f"img/im_{step:06d}.png"
  plt.imsave(filename, grid)
  update_grid(grid)

cmd = f"ffmpeg -y -framerate {video_frame_rate} -pattern_type glob -i '{tmp_dir}/*.png' -vf scale={grid_shape[0]*video_size_factor}:{grid_shape[1]*video_size_factor}:flags=neighbor -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -movflags +faststart out.mp4"
res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.rmtree(tmp_dir, ignore_errors=True)
