import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataloader.pancreas import Pancreas
import os
from medpy import metric
import time
import logging

def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None,
                  preproc_fn=None, AMC=False, print_result=False):
    dc_list = []
    jc_list = []
    hd95_list = []
    asd_list = []
    cnt = 0

    total_num = len(image_list)
    for image_path in image_list:

        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])

        if print_result:
            logging.info('[{}/{}] Sample: {}\ndice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                    .format(cnt+1, total_num, image_path, single_metric[0]*100,single_metric[1]*100,single_metric[2],single_metric[3]))

        dc_list.append(single_metric[0])
        jc_list.append(single_metric[1])
        hd95_list.append(single_metric[2])
        asd_list.append(single_metric[3])

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + str(cnt) + "_gt.nii.gz")
        cnt += 1

    dc_arr = 100 * np.array(dc_list)
    jc_arr = 100 * np.array(jc_list)
    hd95_arr = np.array(hd95_list)
    asd_arr = np.array(asd_list)

    dice_mean = np.mean(dc_arr)
    dice_std = np.std(dc_arr)

    jc_mean = np.mean(jc_arr)
    jc_std = np.std(jc_arr)

    hd95_mean = np.mean(hd95_arr)
    hd95_std = np.std(hd95_arr)

    assd_mean = np.mean(asd_arr)
    assd_std = np.std(asd_arr)

    avg_metric = [dice_mean, jc_mean, hd95_mean, assd_mean]
    std_metric = [dice_std, jc_std, hd95_std, assd_std]

    return avg_metric, std_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, AMC=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                if AMC:
                    net_out = net(test_patch)
                    y1 = (net_out[0] + net_out[1] + net_out[2] + net_out[3]) / 4
                else:
                    y1 = net(test_patch)

                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def test_calculate_metric_Pancreas_AMC(net, test_dataset, num_classes=2, save_result=False, print_result=False, test_save_path='./save'):
    net.eval()
    image_list = test_dataset.image_list

    if save_result:
        test_save_path = Path(test_save_path)
        test_save_path.mkdir(exist_ok=True)

    avg_metric, std_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=4,
                               save_result=save_result, print_result=print_result,
                               test_save_path=str(test_save_path) + '/', AMC=True)
    return avg_metric, std_metric


def test_calculate_metric_LA_AMC(net, test_dataset, num_classes=2, save_result=False, print_result=False, test_save_path='./save'):
    net.eval()

    image_list = test_dataset.image_list
    avg_metric, std_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=save_result, print_result=print_result,
                               test_save_path=test_save_path, AMC=True)
    return avg_metric, std_metric

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)


if __name__ == '__main__':
    pass
