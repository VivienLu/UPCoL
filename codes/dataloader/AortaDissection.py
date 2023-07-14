import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose
import random
import torchio as tio


class AortaDissection(Dataset):
    """ Aorta Dissection Dataset """

    def __init__(self, data_dir, list_dir, split, aug_times=1,
                       scale_range = (0.9,1.1), 
                       rotate_degree = 10):

        self.data_dir = data_dir
        self.list_dir = list_dir
        self.split = split
        self.scale_range = scale_range
        self.rotate_degree = rotate_degree
        self.aug_times = aug_times
        
        if split == 'lab':
            data_path = os.path.join(list_dir,'train_lab.txt')
            self.transform = True
        elif split == 'unlab':
            data_path = os.path.join(list_dir,'train_unlab.txt')
            self.transform = False
        elif split == 'train':
            data_path = os.path.join(list_dir,'train.txt')
            self.transform = True
        else:
            data_path = os.path.join(list_dir,'test.txt')
            self.transform = False

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        self.image_list = [os.path.join(self.data_dir, item) for item in self.image_list]

        print("{} set: total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        if self.transform:
            subject = tio.Subject(image=tio.ScalarImage(tensor=image[np.newaxis,]), \
                label=tio.LabelMap(tensor=label[np.newaxis,]))
            RandomAffine = tio.RandomAffine(scales=self.scale_range,degrees=self.rotate_degree)
            randaff_sample = RandomAffine(subject)
            image =  randaff_sample['image']['data']
            label =  torch.unsqueeze(randaff_sample['label']['data'],0)
        else:
            image = torch.from_numpy(image[np.newaxis,])
            label = torch.from_numpy(label)

        return {'image':image.float(), 'label':label.squeeze().long(), 'name':image_path}


if __name__ == '__main__':

    data_dir = '../../../Datasets/TBAD128'
    list_dir = '../datalist/AD/AD_0'

    labset = AortaDissection(data_dir, list_dir,split='lab')
    unlabset = AortaDissection(data_dir,list_dir,split='unlab')
    trainset = AortaDissection(data_dir,list_dir,split='train')
    testset = AortaDissection(data_dir, list_dir,split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]
 
    print(len(trainset), lab_sample['image'].shape, lab_sample['label'].shape) # 100 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(labset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 20 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(unlabset), train_sample['image'].shape, train_sample['label'].shape) # 80 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 24 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])

    labset = AortaDissection(data_dir, list_dir,split='lab', aug_times=5)
    unlabset = AortaDissection(data_dir,list_dir,split='unlab', aug_times=5)
    trainset = AortaDissection(data_dir,list_dir,split='train', aug_times=5)
    testset = AortaDissection(data_dir, list_dir,split='test', aug_times=5)

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]

    print(len(trainset), lab_sample['image'].shape, lab_sample['label'].shape) # 500 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(labset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 100 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(unlabset), train_sample['image'].shape, train_sample['label'].shape) # 400 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 24 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])