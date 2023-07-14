import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import cv2
import re
import copy
from PIL import Image
import matplotlib.cm as mpl_color_map


f = plt.figure(figsize=(20,7),dpi =500)
name = 'E2ZMO66WGS74UKXTZPPQ'
# name = 'A4R1S23KR0KU2WSYHK2X'
load_path = '../results/LA/testlog/'


iter_list = ['iter_'+str(i) for i in range(400,10000,1200)]
model_paths = [load_path + name + '/RM_' + iter_num for iter_num in iter_list]
cmap = matplotlib.colors.ListedColormap(["gray", "red"])
plt.subplots_adjust(wspace = 0.03, hspace = -0.35)

ax = 1

slice_ind_3 = 40
slice_ind_2 = 56
slice_ind_1 = 50


img_alpha = 0.99
label_alpha = 0.6
img = np.load(load_path + name + '/Img.npy',allow_pickle=True).squeeze()
label = np.load(load_path + name + '/Lab.npy',allow_pickle=True).squeeze()


for model_path in model_paths:
    RM = np.load(model_path+'.npy',allow_pickle=True).squeeze()
    reliable_map = RM    

    rm_3 = reliable_map[:,:,slice_ind_3]
    f.add_subplot(3,len(model_paths)+1,ax); ax+=1
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(img[:,:,slice_ind_3],alpha=img_alpha,cmap='gray')
    plt.imshow(rm_3, alpha=label_alpha, cmap='jet')
    

    rm_2 = reliable_map[:,slice_ind_2,:]
    f.add_subplot(3,len(model_paths)+1,ax+len(model_paths))
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(np.rot90(img[:,slice_ind_2,:]),alpha=img_alpha,cmap='gray')
    plt.imshow(np.rot90(rm_2), alpha=label_alpha, cmap='jet')

    rm_1 = reliable_map[slice_ind_1,:,:]
    f.add_subplot(3,len(model_paths)+1,ax+2*(len(model_paths)+1)-1)
    plt.xticks([]) 
    plt.yticks([]) 
    model = 'iter.'+model_path.split('/')[-1].split('_')[-1]
    plt.xlabel(model,fontsize=14)
    
    plt.imshow(np.rot90(img[slice_ind_1,:,:]),alpha=img_alpha,cmap='gray')
    plt.imshow(np.rot90(rm_1), alpha=label_alpha, cmap='jet')

'''colorbar'''
# cax = plt.axes([-0.2, 0.18, 0.02, 0.6])
# cbar = plt.colorbar(cax=cax)
# cbar.set_ticks([])

plt.savefig(load_path + name + '/RM-test-1.png',bbox_inches='tight',dpi=f.dpi,pad_inches=0.0)
