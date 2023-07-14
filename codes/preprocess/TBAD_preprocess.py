import numpy as np
import os
import random
import nibabel as nib
import traceback
from pathlib import Path
import io_
import matplotlib.pyplot as plt
from preprocess_utils import resample_volume_nib
from preprocess_utils import *
from tqdm import tqdm
import cv2
import re
import h5py
import glob
import torchio as tio
import csv
random.seed(1337)

#########################################################################################
#                                    Data Preprocess                                    #
#########################################################################################

def DatasetPreprocess(src, dst, record_path, mode='9th-TypeB', clip_range=[-401.5, 928.5], target_spacing=(1, 1, 1), resample=True, clip=True, target_shape=(512,512,512)):
    
    """
    :param src: source dir 
    :param dst: tager dir to save
    :param mode: type of the dataset to be processed ['9th-TypeB','ImageTBAD']
    :param clip_range: target range of image clip
    :param target_spacing: target size of image spacing
    """

    try:
        print("Processing...")
        save_dir = os.path.join(dst,mode)
            
        # ------------- Dataset: 9th-TypeB -------------#

        if mode == '9th-TypeB':
            sample_path_list = [os.path.join(src,f) for f in os.listdir(src)]
            print(sample_path_list)
            if resample:
                save_dir = os.path.join(save_dir,'wResample')
                os.makedirs(save_dir, exist_ok=True)
                print('w Resample...')
            else:
                save_dir = os.path.join(save_dir,'woResample')
                os.makedirs(save_dir, exist_ok=True)  
                print('w/o Resample')

            for sample_path in sample_path_list:
                print("Load sample from: ", sample_path) 
                img_path = os.path.join(sample_path,'origin.nii.gz')
                img, spacing, affine_pre = io_.read_nii(img_path)

                # pseudo lummen labeled as 1
                label_path = os.path.join(sample_path,'pseudo.nii.gz')
                mask, _, _ = io_.read_nii(label_path)
                p_mask = mask.astype(int)

                # true lumen labeled as 2
                label_path = os.path.join(sample_path,'true.nii.gz')
                mask, _, _ = io_.read_nii(label_path)
                t_mask = mask.astype(int) * 2

                if t_mask.shape != p_mask.shape:
                    print("wrong!")
                    continue
                mask = t_mask + p_mask
                # label the pixel labeled as TL and FL at the same time as background
                mask[mask == 3] = 0

                # assert mask.shape == img.shape, "{}, {}".format(mask.shape, img.shape)÷
                if mask.shape != img.shape:
                    print("wrong image shape", sample_path)
                    continue

                # resample to [1, 1, 1] change the pixel size and depth
                if resample:
                    spacing = (spacing[1], spacing[1], spacing[1])
                    affine_pre = io_.make_affine2(spacing)
                    processed_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
                    processed_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)
                else:
                    processed_img, processed_mask = img, mask

                resampled_img, resampled_mask = processed_img, processed_mask

                # # clip to [-401.5, 928.5] with window range of 1327 and window level of 265
                if clip:
                    min_clip, max_clip = clip_range[0], clip_range[1]
                    processed_img = processed_img.clip(min_clip, max_clip)
                    processed_img = normalize(processed_img)
                
                # if CropOrPad:
                subject = tio.Subject(image=tio.ScalarImage(tensor=processed_img[np.newaxis,]), \
                                label=tio.LabelMap(tensor=processed_mask[np.newaxis,]))
                Resize = tio.Resize(target_shape)
                Resized = Resize(CropOrPad(subject))
                processed_img = Resized['image']['data'].squeeze().cpu().numpy()
                processed_mask = Resized['label']['data'].squeeze().cpu().numpy()

                case_idx = os.path.basename(sample_path)
                # save_to_h5(processed_img, processed_mask, os.path.join('/data/luwenjing/programmes/MultiStageSeg/Test/', case_idx + '.h5'))
                print('saved : {}, original shape : {}, processed shape : {}'.format(case_idx, img.shape, processed_mask.shape))
                # show_graphs(img[:,-256:-240,:],processed_img[:,-256:-240,:],processed_mask[:,-256:-240,:], \
                #                 (16,32),'/data/luwenjing/programmes/MultiStageSeg/Test/'+case_idx+'_img2.png')
                mid_slice = round(resampled_img.shape[1]/2)
                show_graphs_test(img[:,-272:-240,:],resampled_img[:,mid_slice-16:mid_slice+16,:],processed_img[:,-272:-240,:],mask[:,-272:-240,:],processed_mask[:,-272:-240,:], \
                                (25,64),'../../../../Datasets/Resize9thTypeB/'+case_idx+'_img6.png')                 

        # ------------- Dataset: ImageTBAD -------------#

        else: # mode == 'ImageTBAD'
            os.makedirs(save_dir, exist_ok=True)
            sample_path_dir = os.path.join(src,'*image.nii.gz')
            sample_path_list = [patient for patient in glob.glob(sample_path_dir)][30:]
            num = 0
            total_num = len(sample_path_list)
            for sample_path in sample_path_list:
                num += 1
                print("[{}/{}]\tload sample from: {}".format(num,total_num,sample_path))
                img, spacing, affine_pre = io_.read_nii(sample_path)

                label_path = re.sub('image', 'label', sample_path)
                label, _, _ = io_.read_nii(label_path)
                label = label.astype(int)
                
                # label the pixel labeled as TFL as background
                label[label == 3] = 0

                # pseudo lummen labeled as 1 and true lumen labeled as 2
                label_TL, label_FL = label.copy(), label.copy()
                label_TL[label_TL != 1] = 0 # label of true lumen in ImageTBAD dataset is 1 
                label_FL[label_FL != 2] = 0 # label of false lumen in ImageTBAD dataset is 2
                mask = label_FL/2 + label_TL*2

                # assert mask.shape == img.shape, "{}, {}".format(mask.shape, img.shape)÷
                if mask.shape != img.shape:
                    print("wrong image shape", case_folder)
                    continue

                # resample to [1, 1, 1] change the pixel size and depth
                if resample:
                    spacing = (spacing[1], spacing[1], spacing[1])
                    affine_pre = io_.make_affine2(spacing)
                    processed_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
                    processed_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)
                else:
                    processed_img, processed_mask = img, mask

                resampled_img, resampled_mask = processed_img, processed_mask
                # clip to clip_range 
                if clip:
                    min_clip, max_clip = clip_range[0], clip_range[1]
                    processed_img = processed_img.clip(min_clip, max_clip)
                processed_img = normalize(processed_img)

                subject = tio.Subject(image=tio.ScalarImage(tensor=processed_img[np.newaxis,]), \
                                label=tio.LabelMap(tensor=processed_mask[np.newaxis,]))
                Resize = tio.Resize(target_shape)
                Resized = Resize(subject)
                processed_img = Resized['image']['data'].squeeze().cpu().numpy()
                processed_mask = Resized['label']['data'].squeeze().cpu().numpy()

                case_idx = 'TBAD-' + os.path.basename(sample_path).split('_')[0]
                saved_path = os.path.join(save_dir, case_idx + '.h5')
                Resample_mid_slice2 = int(round(resampled_img.shape[-1]/2))
                Resample_mid_slice1 = int(round(resampled_img.shape[1]/2))
                mid_slice = 256

                print('Name: {}\nRaw size: {}\nResample size: {}\nPad size: {}\nSave at: {}' \
                        .format(case_idx,img.shape,resampled_img.shape,processed_img.shape,saved_path))
                # save_to_h5(processed_img, processed_mask, os.path.join(save_dir, case_idx + '.h5'))
                show_graphs_test(img[:,-272:-240,:], \
                                resampled_img[:,Resample_mid_slice1-16:Resample_mid_slice1+16,:], \
                                processed_img[:,-272:-240,:], \
                                label[:,-272:-240,:], \
                                processed_mask[:,-272:-240,:], \
                                (25,64), \
                                record_path+'/img/'+case_idx+'_img1.png', \
                                dim=1)                
                show_graphs_test(img[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                                resampled_img[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                                processed_img[:,:,mid_slice-16:mid_slice+16], \
                                label[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                                processed_mask[:,:,mid_slice-16:mid_slice+16], \
                                (16,80), \
                                record_path+'/img/'+case_idx+'_img2.png', \
                                dim=2)           
                # save and record
                save_to_h5(processed_img, processed_mask, saved_path)
                csv_file = open(record_path+'/Processed_ImageTBAD.csv', 'a+', newline='', encoding='gbk')
                writer = csv.writer(csv_file)
                # writer.writerow(['Path', 'Saved Path', 'Raw shape','Processed shape'])
                writer.writerow([sample_path,saved_path,str(img.shape),str(processed_img.shape)])
                csv_file.close()
        
    except Exception as e:
        print("Error:")
        print(e)
        traceback.print_exc()

'''
Generate dataset for stage 1:
    To accelerate the speed of loading data
    Resize the preprocessed data to target size and save
'''
def Generate_Stage1_Dataset(data_path, save_path, target_size):

    data_list = [os.path.join(data_path,f) for f in os.listdir(data_path)]

    num = 0
    total_num = len(data_list)

    for image_path in data_list:
        # load data
        num += 1
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:][np.newaxis,np.newaxis,], h5f['label'][:].astype(np.float32)[np.newaxis,np.newaxis,]
        # resize to target_size
        image_new = F.interpolate(torch.from_numpy(image),size = target_size).cpu().numpy().squeeze()
        label_new = F.interpolate(torch.from_numpy(label),size = target_size).cpu().numpy().squeeze()
        # save new file
        image_path_new = re.sub(data_path,save_path,image_path)
        os.makedirs(os.path.dirname(image_path_new), exist_ok=True)
        save_to_h5(image_new,label_new,image_path_new)
        print('[{}/{}] {}%\nOriginal path: {}\tSave path: {}\nraw image: {}, new image: {}\n'
                .format(num,total_num,round(num / total_num *100),image_path,image_path_new,image.shape,image_new.shape))

'''
Another choice for preprocessing each sample in 9th TypeB:
    1. Make sure the mask range in each dimension
    2. Plot the slice in differernt views
    3. Crop the target range of CT
    4. Pad the hight and width to the taget size
    4. Resize to the total sample to the target size
'''
def DatasetPreprocessEach(src, dst, record_path, target_spacing, clip_range):
    # load data
    img_path = os.path.join(src,'origin.nii.gz')
    img, spacing, affine_pre = io_.read_nii(img_path)

    fl_path = os.path.join(src,'pseudo.nii.gz')
    fl_mask, _, _ = io_.read_nii(fl_path)
    p_mask = fl_mask.astype(int)

    tl_path = os.path.join(src,'true.nii.gz')
    tl_mask, _, _ = io_.read_nii(tl_path)
    t_mask = tl_mask.astype(int) * 2

    mask = t_mask + p_mask
    mask[mask == 3] = 0

    # resample data
    spacing = (spacing[1], spacing[1], spacing[1])
    affine_pre = io_.make_affine2(spacing)
    resampled_img, affine = resample_volume_nib(img, affine_pre, spacing, target_spacing, mask=False)
    resampled_mask, affine = resample_volume_nib(mask, affine_pre, spacing, target_spacing, mask=True)

    # compute range
    bbox = get_bbox_3d(resampled_mask)
    channel_range = np.array(bbox[0])

    # clip the image
    min_clip, max_clip = clip_range[0], clip_range[1]
    clipped_img = resampled_img.clip(min_clip, max_clip)
    clipped_img = normalize(resampled_img)

    # crop and resize the image
    # processed_img, processed_mask = clipped_img, resampled_mask
    # mid_slice = int(round(np.mean(channel_range)))

    # clipped_img, resampled_mask = clipped_img[:,:,channel_range[0]-25:channel_range[1]+50], resampled_mask[:,:,channel_range[0]-25:channel_range[1]+50]
    clipped_img, resampled_mask = clipped_img[:,:,channel_range[0]-25:], resampled_mask[:,:,channel_range[0]-25:]
    subject = tio.Subject(image=tio.ScalarImage(tensor=clipped_img[np.newaxis,]), \
                                label=tio.LabelMap(tensor=resampled_mask[np.newaxis,]))
    Resize = tio.Resize((512,512,512))
    Resized_sub = Resize(subject)
    processed_img = Resized_sub['image']['data'].squeeze().cpu().numpy()
    processed_mask = Resized_sub['label']['data'].squeeze().cpu().numpy()
    mid_slice = 256

    # show
    Resample_mid_slice2 = int(round(resampled_img.shape[-1]/2))
    Resample_mid_slice1 = int(round(resampled_img.shape[1]/2))
    case_idx = os.path.basename(src)
    saved_path = os.path.join(dst, case_idx + '.h5')
    print('Name: {}\nRaw size: {}\nResample size: {}\nPad size: {}\nMask range: {}\nmid slice: {}\nSave at: {}' \
            .format(case_idx,img.shape,resampled_img.shape,processed_img.shape,channel_range,mid_slice,saved_path))
    show_graphs_test(img[:,-272:-240,:], \
                     resampled_img[:,Resample_mid_slice1-16:Resample_mid_slice1+16,:], \
                     processed_img[:,-272:-240,:], \
                     mask[:,-272:-240,:], \
                     processed_mask[:,-272:-240,:], \
                     (25,64), \
                     record_path+'/img/'+case_idx+'_img1.png', \
                     dim=1)                
    show_graphs_test(img[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                     resampled_img[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                     processed_img[:,:,mid_slice-16:mid_slice+16], \
                     mask[:,:,Resample_mid_slice2-16:Resample_mid_slice2+16], \
                     processed_mask[:,:,mid_slice-16:mid_slice+16], \
                     (16,80), \
                     record_path+'/img/'+case_idx+'_img2.png', \
                     dim=2)  

    # save and record
    save_to_h5(processed_img, processed_mask, saved_path)
    csv_file = open(record_path+'/Processed_9thTypeB.csv', 'a+', newline='', encoding='gbk')
    writer = csv.writer(csv_file)
    # writer.writerow(['Path', 'Saved Path', 'Raw shape','Resample shape','Mask Box range','Crop Channel range'])
    writer.writerow([src,saved_path,str(img.shape),str(resampled_img.shape),str(channel_range),str([channel_range[0]-25,])])
    csv_file.close()

def normalize(data, norm_type = 'scale'):
    if norm_type == 'z-score':
        normalized_data = (data - data.mean()) / (data.std() + 1e-10)
    else:
        normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data

def save_to_h5(img, mask, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('image', data=img)
    hf.create_dataset('label', data=mask)
    hf.close()


#########################################################################################
#                                 Debug / Visualization                                 #
#########################################################################################

def show_graphs(raw_img,img,mask,FigSize,savename):
    plt.figure(figsize=FigSize)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    range_show = img.shape[1]
    for i in range(range_show):
        plt.subplot(range_show,3,i*3+1)
        plt.imshow(raw_img[:,i,:])
        plt.subplot(range_show,3,i*3+2)
        plt.imshow(img[:,i,:])
        plt.subplot(range_show,3,i*3+3)
        plt.imshow(mask[:,i,:])
    plt.savefig(savename)

def show_graphs_test(raw_img,resample_img,img,raw_mask,mask,FigSize,savename,dim):
    plt.figure(figsize=FigSize)
    range_show = img.shape[dim]
    if dim == 1:
        for i in range(range_show):
            plt.subplot(range_show,5,i*5+1)
            plt.imshow(raw_img[:,i,:])
            plt.subplot(range_show,5,i*5+2)
            plt.imshow(resample_img[:,i,:])
            plt.subplot(range_show,5,i*5+3)
            plt.imshow(img[:,i,:])
            plt.subplot(range_show,5,i*5+4)
            plt.imshow(raw_mask[:,i,:])
            plt.subplot(range_show,5,i*5+5)
            plt.imshow(mask[:,i,:])
    elif dim == 2:
        for i in range(range_show):
            plt.subplot(range_show,5,i*5+1)
            plt.imshow(raw_img[:,:,i])
            plt.subplot(range_show,5,i*5+2)
            plt.imshow(resample_img[:,:,i])
            plt.subplot(range_show,5,i*5+3)
            plt.imshow(img[:,:,i])
            plt.subplot(range_show,5,i*5+4)
            plt.imshow(raw_mask[:,:,i])
            plt.subplot(range_show,5,i*5+5)
            plt.imshow(mask[:,:,i])
    plt.savefig(savename)


if __name__ == "__main__":

    '''
    Preprocess all samples at the same time:
    1. create a csv file to record
    2. process the single dataset
    '''
    # root0 = '../../../../dataset/AD/type_2'
    # root1 = '../../../../dataset/ImageTBAD'
    # dst = '../../../../Datasets/TBAD/'
    # target_shape = (128,128,128)

    # csv_file = open(record_path+'/Processed_ImageTBAD.csv', 'w', newline='', encoding='gbk')
    # writer = csv.writer(csv_file)
    # writer.writerow(['Path', 'Saved Path', 'Raw shape','Processed shape'])
    # csv_file.close()

    # record_path = '../../../../Datasets/9th-TypeB'
    # DatasetPreprocess(root0,dst,record_path, mode='9th-TypeB',resample=True,clip=True)

    # record_path = '../../../../Datasets/ImageTBAD'
    # DatasetPreprocess(root1,dst,record_path, mode='ImageTBAD',resample=False,clip=False)

    '''
    Preprocess each sample in single dataset:
    1. create a csv file to record
    2. choose the idex of the samples to be processed and save the middle slices in 2 different views
    3. according to the visualization to crop the images and record
    '''
    # idx = 23
    # src_list = os.listdir(root0)
    # src_test = os.path.join(root0,src_list[idx])

    # csv_file = open(record_path+'/Processed_9thTypeB.csv', 'w', newline='', encoding='gbk')
    # writer = csv.writer(csv_file)
    # writer.writerow(['Path', 'Saved Path', 'Raw shape','Resample shape','Mask Box range','Crop Channel range'])
    # csv_file.close()
    # DatasetPreprocessEach(src_test, dst, record_path, (1,1,1), [-401.5, 928.5])
