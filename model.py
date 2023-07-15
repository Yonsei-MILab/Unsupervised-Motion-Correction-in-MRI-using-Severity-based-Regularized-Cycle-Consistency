import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SN


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        commonmodel = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        self.commonmodel = nn.Sequential(*commonmodel)

        # correction branch  
        # Downsampling
        correctionmodel = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            correctionmodel += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        for _ in range(n_residual_blocks):
            correctionmodel += [ResidualBlock(in_features)]
        out_features = in_features//2
        for _ in range(2):
            correctionmodel += [  nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1, output_padding=0),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        correctionmodel += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.correctionmodel = nn.Sequential(*correctionmodel)

        # estimation branch
        estimationmodel = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            estimationmodel += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        estimationmodel += [ResidualBlock(in_features)]
        estimationmodel += [ResidualBlock(in_features)]
        estimationmodel += [ResidualBlock(in_features)]
        estimationmodel += [  nn.Conv2d(256, 128, 4, stride=4, padding=0), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        estimationmodel += [  nn.Conv2d(128, 64, 4, stride=4, padding=0), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        estimationmodel += [  nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True) ]
        self.estimationmodel = nn.Sequential(*estimationmodel)
        self.linear0 = nn.Sequential(nn.Linear(256,5))


    def forward(self, x):
        return self.correctionmodel(self.commonmodel(x)), self.linear0(self.estimationmodel(self.commonmodel(x)).view(-1,256))


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


def conv2d(params_list, batch_norm = True):
    channel_in, channel_out, kernel_size, stride, padding, activation = params_list
    layers = []
    if batch_norm:
        layers += [SN(nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)),
                   nn.BatchNorm2d(channel_out)]
        nn.init.xavier_uniform_(layers[0].weight)
    else:
        layers += [SN(nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False))]
        nn.init.xavier_uniform_(layers[0].weight)
        
    if activation.lower() == 'relu':
        layers += [nn.ReLU(inplace=True)]
    if activation.lower() == 'leakyrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    if activation.lower() == 'tanh':
        layers += [nn.Tanh()]
    if activation.lower() == 'sigmoid':
        layers += [nn.Sigmoid()]
        
    return nn.Sequential(*layers)


gfc = 64
dfc = 64
ndfc = 32


    
class Discriminator(nn.Module):
    def __init__(self, input_c, output_c):
        
        cfg_d_K = [[input_c, dfc, 4, 2, 1, 'LeakyReLU'], 
                [dfc, 2*dfc, 4, 2, 1, 'LeakyReLU'], 
                [2*dfc, 4*dfc, 4, 2, 1, 'LeakyReLU'],
                [4*dfc, 8*dfc, 4, 1, 1, 'LeakyReLU'],
                [8*dfc, output_c, 4, 1, 1, 'Sigmoid']]

        super(Discriminator, self).__init__()
        
        self.conv1 = conv2d(cfg_d_K[0], batch_norm=False)
        self.conv2 = conv2d(cfg_d_K[1])
        self.conv3 = conv2d(cfg_d_K[2])
        self.conv4 = conv2d(cfg_d_K[3])
        self.conv5 = conv2d(cfg_d_K[4], batch_norm=False)
        
        
    def forward(self, x):
        
        output = self.conv1(x)
        output = self.conv2(output)
        output = F.pad(output, (0,1,0,1), mode='replicate')
        output = self.conv3(output)
        output = F.pad(output, (0,1,0,1), mode='replicate')
        output = self.conv4(output)
        output = F.pad(output, (0,1,0,1), mode='replicate')
        output = self.conv5(output)
        
        return output


        