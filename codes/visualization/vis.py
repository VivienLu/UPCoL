import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import cv2
import re



f = plt.figure(figsize=(20,4),dpi =500)
name = '5BHTH9RHH3PQT913I'
# name = 'A4R1S23KR0KU2WSYHK2X'
# name = 'E2ZMO66WGS74UKXTZPPQ'

load_path = '../results/LA/showlog/'

iter_list = ['iter_'+str(i) for i in range(400,10000,1200)]
model_paths = [+ name + '/Prot_' + iter_num for iter_num in iter_list]
labels=["Left Atrium"]
cp = ["red"]
cmap = matplotlib.colors.ListedColormap(["gray", "red"])
plt.subplots_adjust(wspace = 0.03, hspace = -0.32)

ax = 1

slice_ind_3 = 40
slice_ind_2 = 40
slice_ind_1 = 74


img_alpha = 0.99
label_alpha = 0.6
img = np.load(load_path + name + '/Img.npy',allow_pickle=True).squeeze()
label = np.load(load_path + name + '/Lab.npy',allow_pickle=True).squeeze()


for model_path in model_paths:

    prot_pred = np.load(model_path+'.npy',allow_pickle=True).squeeze()
    seg_pred= np.load(re.sub('Prot','Seg',model_path)+'.npy',allow_pickle=True).squeeze()

    prot = prot_pred[:,slice_ind_2,:]
    f.add_subplot(2,len(model_paths)+1,ax); ax+=1
    plt.xticks([]) 
    plt.yticks([]) 
    model = model_path.split('/')[-1]
    if ax == 2:
        plt.ylabel('Prototype-based                        \npredictions                       ',fontsize=14, rotation=0)
    
    plt.imshow(np.rot90(img[:,slice_ind_2,:]),alpha=img_alpha,cmap='gray')
    plt.imshow(np.rot90(prot.astype(np.uint8)), alpha=label_alpha, vmin=0, vmax=len(cmap.colors),cmap=cmap)

    seg = seg_pred[:,slice_ind_2,:]
    f.add_subplot(2,len(model_paths)+1,ax+len(model_paths))
    plt.xticks([]) 
    plt.yticks([]) 
    model = 'iter.' + model_path.split('/')[-1][10:]
    plt.xlabel(model,fontsize=14)
    if ax == 2:
        plt.ylabel('Model output                       \npredictions                       ',fontsize=14, rotation=0)

    plt.imshow(np.rot90(img[:,slice_ind_2,:]),alpha=img_alpha,cmap='gray')
    plt.imshow(np.rot90(seg.astype(np.uint8)), alpha=label_alpha, vmin=0, vmax=len(cmap.colors),cmap=cmap)

f.add_subplot(2,len(model_paths)+1,ax); ax+=1
plt.xticks([]) 
plt.yticks([]) 

plt.imshow(np.rot90(img[:,slice_ind_2,:]),alpha=img_alpha,cmap='gray')
plt.imshow(np.rot90(label[:,slice_ind_2,:]),alpha=label_alpha,vmin=0, vmax=len(cmap.colors),cmap=cmap)

f.add_subplot(2,len(model_paths)+1,ax+len(model_paths))
plt.xticks([]) 
plt.yticks([]) 
plt.xlabel('Ground Truth',fontsize=14)

plt.imshow(np.rot90(img[:,slice_ind_2,:]),alpha=img_alpha,cmap='gray')
plt.imshow(np.rot90(label[:,slice_ind_2,:]),alpha=label_alpha,vmin=0, vmax=len(cmap.colors),cmap=cmap)


plt.savefig('../results/LA/showlog/' + name + '/Unlab-2.png',bbox_inches='tight',dpi=f.dpi,pad_inches=0.0)
