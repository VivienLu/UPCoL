# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import os
from datetime import datetime
import pytz
import torch
import shutil

__all__ = ['get_cur_time','checkpoint_save','checkpoint_load']

def get_cur_time():
    return datetime.strftime(datetime.now(pytz.timezone('Asia/Shanghai')), '%Y-%m-%d_%H-%M-%S')

def checkpoint_save(model, is_best,name):
    if is_best:
        torch.save(model.state_dict(), os.path.join(name, 'checkpoint.pth'))
        print('Saved checkpoint:', os.path.join(name, 'checkpoint.pth'))

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch
