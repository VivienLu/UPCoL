import os
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm

from dataloader.LeftAtrium import LAHeart
from dataloader.pancreas import Pancreas
from dataloader.AortaDissection import AortaDissection

import utils.loss
from utils import statistic, ramps
from utils.loss import DiceLoss, SoftIoULoss, to_one_hot
from utils.losses import FocalLoss
from utils.logger import get_cur_time,checkpoint_save
from utils.Generate_Prototype import *
from utils.train_util import *

from networks.vnet_AMC import VNet_AMC

import logging
import sys
import argparse
import re
import shutil


def get_arguments():

    parser = argparse.ArgumentParser(description='Semi-supervised Learning for LA')

    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default='../results')
    parser.add_argument('--load_ema_path', type=str, default='../results')

    # dataset
    parser.add_argument("--data_dir", type=str, default='../../../Datasets/LA_dataset',
                        help="Path to the dataset.")
    parser.add_argument("--list_dir", type=str, default='../datalist/LA',
                        help="Paths to cross-validated datasets, list of test sets and all training sets (including all labeled and unlabeled samples)")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")
    parser.add_argument("--aug_times", type=int, default=1,
                        help="times of augmentation for training.")
    parser.add_argument('--consistency_rampup', type=float,
                        default=300.0, help='consistency_rampup')

    # Optimization options
    parser.add_argument('--lab_batch_size', type=int,  default=1, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int,  default=2, help='batch size')
    parser.add_argument('--lr', type=float,  default=0.001, help='maximum epoch number to train')

    parser.add_argument('--beta1', type=float,  default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float,  default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float,  default=1, help='multiplier of prototype')
    
    # Miscs
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    
    return parser.parse_args()

args = get_arguments()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# create logger
save_path = '../results/LA/showlog'
os.makedirs(save_path,exist_ok = True)

# record
logging.basicConfig(filename=save_path + "/showlog.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('Save at: {}'.format(save_path))

def create_model(ema=False):
    net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def create_dataloader():
    if 'LA' in args.data_dir:
        train_labset = LAHeart(args.data_dir,args.list_dir,split='lab', aug_times = args.aug_times)
        train_unlabset = LAHeart(args.data_dir,args.list_dir,split='unlab', aug_times = args.aug_times)
        testset = LAHeart(args.data_dir,args.list_dir,split='test', aug_times = args.aug_times)
    elif 'TBAD' in args.data_dir:
        train_labset = AortaDissection(args.data_dir,args.list_dir,split='lab', aug_times = args.aug_times)
        train_unlabset = AortaDissection(args.data_dir,args.list_dir,split='unlab', aug_times = args.aug_times)
        testset = AortaDissection(args.data_dir,args.list_dir,split='test', aug_times = args.aug_times)
    else:
        train_labset =Pancreas(args.data_dir, args.list_dir,split='lab', aug_times = args.aug_times)
        train_unlabset = Pancreas(args.data_dir,args.list_dir,split='unlab', aug_times = args.aug_times)
        testset = Pancreas(args.data_dir,args.list_dir,split='test', aug_times = args.aug_times) 
    
    trainlab_loader = DataLoader(train_labset, batch_size=args.lab_batch_size, shuffle=False, num_workers=0)
    trainunlab_loader = DataLoader(train_unlabset, batch_size=args.unlab_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    logging.info("{} iterations for lab per epoch.".format(len(trainlab_loader)))
    logging.info("{} iterations for unlab per epoch.".format(len(trainunlab_loader)))
    logging.info("{} samples for test.\n".format(len(test_loader)))
    return trainlab_loader, trainunlab_loader, test_loader

def load_net_opt_return(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    logging.info('Loaded from {}'.format(path))
    return state['epoch']


def save_npy():
    set_random_seed(args.seed)

    trainlab_loader, trainunlab_loader, test_loader = create_dataloader()

    import glob 
    epoch_list = glob.glob('../results/LA/checkpoints/epoch_*.pth')
    def get_iter(element):
        return int(os.path.basename(element).split('_')[1])

    epoch_list.sort(key=get_iter, reverse=False)

    total_num = len(trainlab_loader)

    for step, (labeled_batch, unlabeled_batch) in enumerate(zip(trainlab_loader,trainunlab_loader)):

        if 'E2ZMO66WGS74UKXTZPPQ' in unlabeled_batch['name'][1]:
            lab_img, lab_lab = labeled_batch['image'].cuda(), labeled_batch['label'].cuda()
            unlab_img, unlab_lab = unlabeled_batch['image'].cuda(), unlabeled_batch['label'].cuda()
            lab_lab_onehot = to_one_hot(lab_lab.unsqueeze(1), args.num_classes)

            lab_sample_name = labeled_batch['name']
            unlab_sample_name = unlabeled_batch['name']

            lab_file_name = os.path.basename(os.path.dirname(unlab_sample_name[1]))
            save_lab_fold = os.path.join(save_path, lab_file_name)
            os.makedirs(save_lab_fold, exist_ok=True)

            img_lab_name = os.path.join(save_lab_fold, 'Img.npy')
            lab_lab_name = os.path.join(save_lab_fold, 'Lab.npy')
            np.save(img_lab_name,unlab_img[1].cpu().numpy())
            np.save(lab_lab_name,unlab_lab[1].cpu().numpy())
            print(unlab_img[1].shape, unlab_lab[1].shape)

            for net_path in epoch_list:
                ema_net_path = re.sub('epoch','ema_epoch',net_path)

                net = create_model().cuda()
                ema_net = create_model(ema=True).cuda()
                optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) 

                epoch = load_net_opt_return(net, optimizer, net_path)
                epoch = load_net_opt_return(ema_net, optimizer, ema_net_path)
                iter_num = os.path.basename(net_path).split('_')[3]

                '''Supervised'''
                lab_out = net(lab_img)

                lab_out_ce = lab_out[0]
                lab_out_dice = lab_out[1]
                lab_out_focal = lab_out[2]
                lab_out_iou = lab_out[3]
                lab_fts = lab_out[4]

                lab_seg = (lab_out_ce + lab_out_dice + lab_out_focal + lab_out_iou) / 4

                # prototypical alignment
                lab_fts = F.interpolate(lab_fts, size=lab_lab.shape[-3:],mode='trilinear')
                lab_prototypes = getPrototype(lab_fts, lab_lab_onehot)

                unlab_ema_out = ema_net(unlab_img)

                unlab_ema_out_pred = (unlab_ema_out[0] + unlab_ema_out[1] + unlab_ema_out[2] + unlab_ema_out[3]) / 4
                unlab_ema_fts = unlab_ema_out[4]

                unlab_ema_out_soft = torch.softmax(unlab_ema_out_pred, dim=1)
                unlab_ema_mask = torch.argmax(unlab_ema_out_soft, dim=1)
                unlab_ema_mask_onehot = to_one_hot(unlab_ema_mask.unsqueeze(1), args.num_classes)

                # uncertainty assesment
                uncertainty =  -torch.sum(unlab_ema_out_soft * torch.log(unlab_ema_out_soft  + 1e-16), dim=1)
                norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty],dim=0)

                reliable_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-3:]))

                unlab_ema_fts = F.interpolate(unlab_ema_fts, size=unlab_ema_mask.shape[-3:],mode='trilinear')
                unlab_prototypes = getPrototype(unlab_ema_fts, unlab_ema_mask_onehot, reliable_map)

                consistency_weight = get_current_consistency_weight(epoch, args.consistency_rampup)

                '''Certainty'''
                prototypes = [ (lab_prototypes[c] + consistency_weight * unlab_prototypes[c]) / (1 + consistency_weight) for c in range(args.num_classes)]

                lab_dist =  [calDist(lab_fts, prototype, scaler=args.scaler) for prototype in prototypes]
                lab_dist = torch.stack(lab_dist, dim=1)
                unlab_dist = [calDist(unlab_ema_fts, prototype, scaler=args.scaler) for prototype in prototypes]
                unlab_dist = torch.stack(unlab_dist, dim=1)

                lab_prot_pred = torch.argmax(lab_dist, dim=1)
                unlab_prot_pred = torch.argmax(unlab_dist, dim=1)


                lab_masks = torch.softmax(lab_seg, dim = 1)
                lab_masks = torch.argmax(lab_masks, dim = 1)

                logging.info('\n[{}/{}]--------------------------------------------------------'.format(step+1, total_num))
                # lab_train_dice = statistic.dice_ratio(lab_masks, lab_lab)
                # lab_prot_dice = statistic.dice_ratio(lab_prot_pred, lab_lab)
                # logging.info('Label sample: {}\tSeg Dice: {:.4f}%\tProt Dice: {:.4f}%'.format(lab_sample_name[0], lab_train_dice*100, lab_prot_dice*100))

                # for i in range(len(unlab_sample_name)):
                #     unlab_train_dice = statistic.dice_ratio(unlab_ema_mask[i], unlab_lab[i])
                #     unlab_prot_dice = statistic.dice_ratio(unlab_prot_pred[i], unlab_lab[i])
                #     logging.info('Unlab sample: {}\tSeg Dice: {:.4f}%\tProt Dice: {:.4f}%'.format(unlab_sample_name[i], unlab_train_dice*100, unlab_prot_dice*100))
                
                unlab_train_dice = statistic.dice_ratio(unlab_ema_mask[1], unlab_lab[1])
                unlab_prot_dice = statistic.dice_ratio(unlab_prot_pred[1], unlab_lab[1])
                logging.info('Unlab sample: {}\tSeg Dice: {:.4f}%\tProt Dice: {:.4f}%'.format(unlab_sample_name[1], unlab_train_dice*100, unlab_prot_dice*100))
               

                seg_lab_name = os.path.join(save_lab_fold, 'Seg_iter_{}.npy'.format(iter_num))
                prot_lab_name = os.path.join(save_lab_fold, 'Prot_iter_{}.npy'.format(iter_num))
                np.save(seg_lab_name,unlab_ema_mask[1].cpu().numpy().squeeze())
                np.save(prot_lab_name,unlab_prot_pred[1].cpu().numpy().squeeze())
                print(unlab_ema_mask[1].squeeze().shape, unlab_prot_pred[1].squeeze().shape)
                rm_lab_name = os.path.join(save_lab_fold, 'RM_iter_{}.npy'.format(iter_num))
                np.save(rm_lab_name,reliable_map[1].cpu().numpy().squeeze())
                print(reliable_map[1].shape)


if __name__ == '__main__':
    '''
    label:  5BHTH9RHH3PQT913I59W
    unlabel:    A4R1S23KR0KU2WSYHK2X
    '''
    save_npy()



