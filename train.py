import os
import os.path as osp
from numpy import True_
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from statistics import mean
import torchvision
from utils import *
from dataset import *
from model import *
from utilss import *

from torchsummary import summary
import random

from IQA_pytorch import SSIM, utils

SSIM1ch = SSIM(channels=1)

class Train:
    def __init__(self, args):

        self.gpu_num = args.gpu_num
        self.sim_mode = args.sim_mode
        self.device = torch.device("cuda:%d"%self.gpu_num) if torch.cuda.is_available() else 'cpu'

        self.vis_num = args.vis_num

        self.save_dir = args.save_dir
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.tsbd_name = args.tsbd_name

        self.bs = args.bs
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.name_data = args.name_data

        self.img_size = args.img_size
        self.img_channel = args.img_channel

        self.beta1 = args.beta1
        self.gamma = args.gamma
        self.step_size = args.step_size

        self.real_label = args.real_label
        self.fake_label = args.fake_label

    def train(self):
        
        gpu_num = self.gpu_num
        sim_mode = self.sim_mode
        save_dir = self.save_dir
        tsbd_name = self.tsbd_name
        name_data = self.name_data
        save_dir = osp.join(save_dir, name_data, tsbd_name)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        bs = self.bs
        lr = self.lr
        num_epochs = self.num_epochs

        img_size = self.img_size
        img_channel = self.img_channel

        beta1 = self.beta1
        gamma = self.gamma
        step_size = self.step_size

        real_label = self.real_label
        fake_label = self.fake_label

        device = self.device

        label_real = torch.full((bs, 1, 32, 32), real_label, dtype=torch.float32, device=device)
        label_fake = torch.full((bs, 1, 32, 32), fake_label, dtype=torch.float32, device=device)


        train_corrupt_loader = torch.utils.data.DataLoader(corruptdataset, batch_size=bs, 
                                                    shuffle=True, drop_last=True)

        train_clear_loader = torch.utils.data.DataLoader(cleardataset, batch_size=bs, 
                                                    shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, 
                                                    shuffle=False, drop_last=True)


        model_G = Generator(img_channel,img_channel).to(device)
        model_D = Discriminator(img_channel,1).to(device)


        criterion_GAN = nn.BCELoss()
        criterion_L1 = nn.L1Loss()
        criterion_param = nn.CrossEntropyLoss()

        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, 0.999))
       
        schedG = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.9)
        schedD = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9)
        
        result = torch.zeros(num_eval,img_channel,img_size,img_size)
        paramresult = torch.zeros(num_eval,5)
        true = torch.zeros(num_eval,img_channel,img_size,img_size)
        plot(eval0[90][0])

        train_start_time = time.time()
        num_iter = 0
        max_iter = num_epochs*len(train_corrupt_loader)

        label_real = torch.full((bs, 1, 32, 32), real_label, dtype=torch.float32, device=device)
        label_fake = torch.full((bs, 1, 32, 32), fake_label, dtype=torch.float32, device=device)

        loss_G_train = 0
        loss_D_train = 0
        loss_C_train = 0
        loss_I_train = 0
        loss_P_train = 0

        writer = SummaryWriter(comment=self.tsbd_name)
        for epoch in range(num_epochs):
            for batch_index, (data_X, motion_param) in enumerate(train_corrupt_loader):
                model_G.train()
                model_D.train()
                
                data_X = data_X.to(device)
                # data_Y = data_Y.to(device)
                data_Y = next(iter(train_clear_loader)).to(device)
                motion_param = motion_param.to(device)
                motion_param = motion_param.view(-1)
                motion_param = motion_param.long()
                model_D.zero_grad()
                output_D_Y_real = model_D(data_Y)
                
                err_D_real = criterion_GAN(output_D_Y_real, label_real)
               
                output_G_Y, pred_param = model_G(data_X)
                
                output_D_Y_fake = model_D(output_G_Y.detach())
                
                err_D_fake = criterion_GAN(output_D_Y_fake, label_fake)
                
                err_D = err_D_real + err_D_fake
                
                err_D.backward()
                
                optimizer_D.step()
                
                model_G.zero_grad()
            
                data_Y_Y,_ = model_G(data_Y)
                err_I = criterion_L1(data_Y, data_Y_Y)

                output_G_Y_real = model_D(output_G_Y)
                err_G = criterion_GAN(output_G_Y_real, label_real)

                if sim_mode == 'proposed':
                    output_G_YtoX = simulator2D(output_G_Y, motion_param, gpu_num=gpu_num)
                    output_G_XtoY, _= model_G(simulator2D(data_Y, motion_param, gpu_num=gpu_num))
                elif sim_mode == 'simonly':
                    output_G_YtoX = simulator2D2(output_G_Y, gpu_num=gpu_num)
                    output_G_XtoY, _= model_G(simulator2D2(data_Y, gpu_num=gpu_num))

                err_C_YtoX = criterion_L1(output_G_YtoX, data_X)
                err_C_XtoY = criterion_L1(output_G_XtoY, data_Y)
                err_C = err_C_YtoX + err_C_XtoY
                
                if sim_mode == 'proposed':
                    err_P = criterion_param(pred_param, motion_param)
                    err_G_C = err_G + 10*err_C + err_P + err_I
                elif sim_mode == 'simonly':
                    err_P = torch.zeros(1)
                    err_G_C = err_G + 10*err_C + err_I

                err_G_C.backward()
                
                optimizer_G.step()
                
                num_iter += 1

                loss_G_train += err_G.item()
                loss_D_train += err_D.item()
                loss_C_train += err_C.item()
                loss_P_train += err_P.item()

                
                if num_iter == 1:
                    avg_SSIM_input = 100*(1-SSIM1ch(eval0,evalfr0))
                    avg_psnr_input, avg_mse_input = psnr256(256*eval0,256*evalfr0)
                    print('Input SSIM:{:.2f}%, psnr:{:.2f}, mse:{:.2f}'.format(avg_SSIM_input, avg_psnr_input, avg_mse_input))


                if num_iter == 10 or num_iter%200 == 0:
                    rand = random.randrange(2,num_eval-1)
                    with torch.no_grad():  
                        for batch_idx, (inputs, motion_param) in enumerate(test_loader):
                            inputs = inputs.to(device)
                            model_G.eval()
                            outputs, evalparam = model_G(inputs)
                            result[batch_idx] = outputs[0]
                            paramresult[batch_idx] = evalparam

                    avg_SSIM_output = 100*(1-SSIM1ch(result,evalfr0))
                    avg_psnr_input, avg_mse_input = psnr256(256*result,256*evalfr0)
                    tbsample = torch.cat([evalfr0[rand].view(-1,1,256,256), eval0[rand].view(-1,1,256,256), result[rand].view(-1,1,256,256), abs(evalfr0[rand].view(-1,1,256,256)-result[rand].view(-1,1,256,256))],0)
                    img_grid = torchvision.utils.make_grid(tbsample, padding=0)
                    writer.add_image(tsbd_name, img_grid, global_step= num_iter)
                    writer.add_scalar('SSIM', avg_SSIM_output, global_step= num_iter)
                    writer.add_scalar('psnr', avg_psnr_input, global_step= num_iter)
                    writer.add_scalar('mse', avg_mse_input, global_step= num_iter)
                                   
                    print('iter[{:04d}/{:04d}] \tLoss_D:{:.4f} \tLoss_G:{:.4f} \tLoss_C:{:.4f} \tLoss_P:{:.4f}\telapsed_time:{:.2f}mins\tSSIM:{:.2f}%\tpsnr:{:.2f}\tmse:{:.2f}'
                            .format(
                            num_iter, max_iter, 
                            loss_D_train/200, 
                            loss_G_train/200, 
                            loss_C_train/200, 
                            loss_P_train/200, (time.time()-train_start_time)/60,
                            avg_SSIM_output, 
                            avg_psnr_input, 
                            avg_mse_input
                            ))
                    loss_G_train = 0
                    loss_D_train = 0
                    loss_C_train = 0
                    loss_P_train = 0

                if num_iter%10000 == 0 or num_iter == max_iter:
                    save_name2 = osp.join(save_dir, 'it{:03d}'.format(num_iter))
                    save_name3 = osp.join(save_dir, 'it{:03d}param'.format(num_iter))
                    
                    torch.save(
                        result, save_name2)
                    torch.save(
                        paramresult, save_name3)

                if num_iter%4000 == 0 or num_iter == max_iter:                        
                    save_name = osp.join(save_dir, 'it{:03d}.net'.format(num_iter))
                    torch.save({
                        'model_G_X': model_G.state_dict()
                    }, save_name)
        
            schedG.step()
            schedD.step()
    
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
    