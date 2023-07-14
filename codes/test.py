import torch
import logging
from medpy import metric
import time
from tqdm import tqdm
import logging
import argparse
import re
import sys
import os
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from networks.vnet_AMC import VNet_AMC

from dataloader.LeftAtrium import LAHeart
from dataloader.pancreas import Pancreas
from dataloader.AortaDissection import AortaDissection

from utils.train_util import *
from utils.test_util import test_calculate_metric_LA_AMC, test_calculate_metric_Pancreas_AMC

def get_arguments():

    parser = argparse.ArgumentParser(description='Semi-supervised Testing for UPCoL: Uncertainty-informed Prototype Consistency Learning for Semi-supervised Medical Image Segmentation')

    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default='../results')

    # dataset
    parser.add_argument("--data_dir", type=str, default='../../../Datasets/LA_dataset',
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str, default='../datalist/LA',
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")

    # Optimization options
    parser.add_argument('--lr', type=float,  default=0.001, help='maximum epoch number to train')
    parser.add_argument('--beta1', type=float,  default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float,  default=0.999, help='params of optimizer Adam')
    
    # Miscs
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    
    return parser.parse_args()


def create_model(args, ema=False):
    net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

@torch.no_grad()
def test_LA_Pancreas(net, val_loader, args, maxdice=0, print_result=False):
    time_start = time.time()
    if 'LA' in args.data_dir:
        avg_metrics, std_metrics = test_calculate_metric_LA_AMC(net, val_loader.dataset, print_result=print_result)
    else:
        avg_metrics, std_metrics = test_calculate_metric_Pancreas_AMC(net, val_loader.dataset, print_result=print_result)
    time_end = time.time()
    val_dice = avg_metrics[0]

    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False

    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))
    
    logging.info('\nDice:')
    logging.info('Mean :%.1f(%.1f)' % (avg_metrics[0], std_metrics[0]))

    logging.info('\nJaccard:')
    logging.info('Mean :%.1f(%.1f)' % (avg_metrics[1], std_metrics[1]))

    logging.info('\nHD95:')
    logging.info('Mean :%.1f(%.1f)' % (avg_metrics[2], std_metrics[2]))

    logging.info('\nASSD:')
    logging.info('Mean :%.1f(%.1f)' % (avg_metrics[3], std_metrics[3]))

    logging.info('Inference time: %.1f' % ((time_end-time_start)/len(val_loader)))

    return val_dice, maxdice, max_flag

@torch.no_grad()
def test_AD(model, data_loader, args, print_result=False, maxdice=0):
    dc_list = []
    jc_list = []
    hd95_list = []
    asd_list = []

    time_start = time.time()
    total_num = len(data_loader)
    for i_batch, sampled_batch in tqdm(enumerate(data_loader)):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.LongTensor).cpu().numpy()

        out = model(volume_batch)
        outputs = (out[0] + out[1] + out[2] + out[3]) / 4
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

        for c in range(1, args.num_classes):
            pred_test_data_tr = outputs.copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label_batch.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            score = cal_metric(pred_gt_data_tr, pred_test_data_tr)
            dc_list.append(score[0])
            jc_list.append(score[1])
            hd95_list.append(score[2])
            asd_list.append(score[3])

            if print_result:
                logging.info('\n[{}/{}] {}%:\t{}'.format(i_batch,total_num,round(i_batch/total_num*100),sampled_batch['name']))
                logging.info('Class: {}: dice {:.1f}% | jaccard {:.1f}% | hd95 {:.1f} | asd {:.1f}'
                                .format(c, score[0]*100,score[1]*100,score[2],score[3]))

    time_end = time.time()  
    dc_arr = 100 * np.reshape(dc_list, [-1, 2]).transpose()
    jc_arr = 100 * np.reshape(jc_list, [-1, 2]).transpose()
    hd95_arr = np.reshape(hd95_list, [-1, 2]).transpose()
    asd_arr = np.reshape(asd_list, [-1, 2]).transpose()

    dice_mean = np.mean(dc_arr, axis=1)
    dice_std = np.std(dc_arr, axis=1)

    jc_mean = np.mean(jc_arr, axis=1)
    jc_std = np.std(jc_arr, axis=1)

    hd95_mean = np.mean(hd95_arr, axis=1)
    hd95_std = np.std(hd95_arr, axis=1)

    assd_mean = np.mean(asd_arr, axis=1)
    assd_std = np.std(asd_arr, axis=1)

    logging.info('Dice Mean: {}, Jaccard Mean: {}, Hd95 Mean: {}, Assd Mean: {}'
                    .format(np.mean(dice_mean), np.mean(jc_mean), np.mean(hd95_mean), np.mean(assd_mean)))
    logging.info('Dice:')
    logging.info('FL :%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    logging.info('TL :%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    logging.info('Mean :%.1f' % np.mean(dice_mean))

    logging.info('\nJaccard:')
    logging.info('FL :%.1f(%.1f)' % (jc_mean[0], jc_std[0]))
    logging.info('TL :%.1f(%.1f)' % (jc_mean[1], jc_std[1]))
    logging.info('Mask :%.1f' % np.mean(jc_mean))

    logging.info('\nHD95:')
    logging.info('FL :%.1f(%.1f)' % (hd95_mean[0], hd95_std[0]))
    logging.info('TL :%.1f(%.1f)' % (hd95_mean[1], hd95_std[1]))
    logging.info('Mask :%.1f' % np.mean(hd95_mean))

    logging.info('\nASSD:')
    logging.info('FL :%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    logging.info('TL :%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    logging.info('Mask :%.1f' % np.mean(assd_mean))

    logging.info('Inference time: %.1f' % ((time_end-time_start)/total_num))

    val_dice = dice_mean.mean()
    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False
    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))

    return val_dice, maxdice, max_flag


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)

def main():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # create logger
    save_path = os.path.join(os.path.dirname(args.load_path),'resultlog')
    os.makedirs(save_path,exist_ok = True)

    # record
    logging.basicConfig(filename=save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Save at: {}'.format(save_path))

    set_random_seed(args.seed)

    net = create_model(args)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    load_net_opt(net, optimizer, Path(args.load_path) / 'best.pth')

    if 'LA' in args.data_dir:
        testset = LAHeart(args.data_dir,args.list_dir,split='test')   
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_LA_Pancreas(net, test_loader, args, print_result=True)
    elif 'TBAD' in args.data_dir:
        testset = AortaDissection(args.data_dir,args.list_dir,split='test')
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_AD(net, test_loader, args, print_result=True)
    else:
        testset = Pancreas(args.data_dir,args.list_dir,split='test')   
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        test_LA_Pancreas(net, test_loader, args, print_result=True)


if __name__ == '__main__':
    main()
    # pass
