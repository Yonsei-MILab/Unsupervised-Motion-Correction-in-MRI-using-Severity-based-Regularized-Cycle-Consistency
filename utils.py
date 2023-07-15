import os
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
 
from utilss import *


class Parser():
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()


    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def write_args(self):
        params_dict = vars(self.__args)

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['name_data'])
        args_name = os.path.join(log_dir, 'args.txt')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)


def psnr256(label, outputs, max_val=256.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    mse = np.mean((img_diff)**2)
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: 
        return 100, mse
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr, mse



def simulator2D(data, 
                motion_param,
                gpu_num = 0
                  ):
    
    std1 = 0.1; std2 = 0.1
    device = torch.device("cuda:%d"%gpu_num)
    scantime = data.shape[2]
    result = torch.zeros(data.shape).to(device)

    trajec = torch.zeros(data.shape[2], 1).to(device)
    traj = torch.as_tensor(carte_traj(trajec, '2D').type(torch.int)).to(device)

    for ii in range(data.shape[0]):
        # index = torch.argmax(motion_param[ii]).item()
        index = motion_param[ii].item()
        if index == 0:
            ms, ran, gae = 1, [1,2], [1,4]
        elif index == 1:
            ms, ran, gae = 3, [1,3], [2,6]
        elif index == 2:
            ms, ran, gae = 5, [1,3], [3,8]
        elif index == 3:
            ms, ran, gae = 7, [1,4], [5,10]
        else:
            ms, ran, gae = 9, [1,5], [7,12]          
        data1 = torch.squeeze(data[ii,0,:,:]).to(device)
        res = torch.tensor([sudden_motion_outlier1(scantime, gae, ran, ms, std1, std2) for _ in range(6)]).to(device)
        motion = torch.transpose(res,1,0).to(device)
        result[ii,0,:,:] = tenmotion2D(data1, motion, traj, device =device,mode='RL').to(device)
    result = abs(result)

    return result.cuda(device = device)



def simulator2D2(data, 
                gpu_num = 0
                  ):
    
    std1 = 0.1; std2 = 0.1
    device = torch.device("cuda:%d"%gpu_num)
    scantime = data.shape[2]
    result = torch.zeros(data.shape).to(device)

    trajec = torch.zeros(data.shape[2], 1).to(device)
    traj = torch.as_tensor(carte_traj(trajec, '2D').type(torch.int)).to(device)

    for ii in range(data.shape[0]):
        index = torch.randint(5, size=(1,)).item()
        if index == 0:
            ms, ran, gae = 1, [1,2], [1,4]
        elif index == 1:
            ms, ran, gae = 3, [1,3], [2,6]
        elif index == 2:
            ms, ran, gae = 5, [1,3], [3,8]
        elif index == 3:
            ms, ran, gae = 7, [1,4], [5,10]
        else:
            ms, ran, gae = 9, [1,5], [7,12]          
        data1 = torch.squeeze(data[ii,0,:,:]).to(device)
        res = torch.tensor([sudden_motion_outlier1(scantime, gae, ran, ms, std1, std2) for _ in range(6)]).to(device)
        motion = torch.transpose(res,1,0).to(device)
        result[ii,0,:,:] = tenmotion2D(data1, motion, traj, device =device,mode='RL').to(device)
    result = abs(result)

    return result.cuda(device = device)
