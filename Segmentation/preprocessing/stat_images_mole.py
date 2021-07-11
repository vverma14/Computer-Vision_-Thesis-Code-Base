"""
   Compute several statistics of the mole dataset.
"""

# Pytorch libraries
import torch
import torchvision
# Classic libraries
import numpy as np
from PIL import Image
import os, sys, glob
import pandas as pd
# perso
sys.path.append("../utils") 
from dataset_module import Mole_dataset_simple
    
# data loader
#------------
path_images = "../../mini_dataset/images_mole/"
path_masks = "../../mini_dataset/images_mole_mask/"
mole_dataset = Mole_dataset_simple(path_images, path_masks)
N = len(mole_dataset)
df = pd.DataFrame(columns=('nameImage','num_loader',
                           'size_x','size_y','prop_seg',
                           'mean_pixel','std_pixel',
                           'mean_pixel_R','std_pixel_R',
                           'mean_pixel_G','std_pixel_G',
                           'mean_pixel_B','std_pixel_B'))
# loop all images
#----------------
for i in range(N):
    print( i )
    # take one image
    img,mask,filename = mole_dataset.__getitem__(i) # load
    mask_binary = (mask>1e-10)
    # stat
    _,ny,nx = img.shape
    df.loc[i] = [filename, i, 
                 nx, ny, mask_binary.sum().item()/(nx*ny),
                 img.mean().item(), img.std().item(),
                 img[0,].mean().item(), img[0,].std().item(),
                 img[1,].mean().item(), img[1,].std().item(),
                 img[2,].mean().item(), img[2,].std().item()]

# save tableau
#-------------
df.to_csv('stat_mole_pixel.csv',index=False)
# stat
# mean pixel R:         # df['mean_pixel_R'].mean() 
# std pixel R:          # df['std_pixel_R'].mean() 
# pct mole:           # df['prop_seg'].mean()
#   -> weight: 
