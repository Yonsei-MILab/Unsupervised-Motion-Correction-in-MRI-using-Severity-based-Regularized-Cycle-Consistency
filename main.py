
import argparse

import torch.backends.cudnn as cudnn
from train import *
from utils import Parser


cudnn.benchmark = True
cudnn.fastest = True

parser = argparse.ArgumentParser(description="Unsupervised Learning for Motion Correction and Assessment in Brain MRI Using Severity-based Regularized Cycle Consistency", 
                                formatter_class =argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--gpu_num", default=0, type=int, dest="gpu_num")

parser.add_argument("--tsbd_name", default='', type=str, dest="tsbd_name")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--bs", default=16, type=int, dest="bs")
parser.add_argument("--num_epochs", default=150, type=int, dest="num_epochs")
parser.add_argument("--save_dir", default='./results', type=str, dest="save_dir")
parser.add_argument('--dir_log', default='./results/log', dest='dir_log')
parser.add_argument("--vis_num", default=2, type=int, dest="vis_num")

parser.add_argument("--img_size", default=256, type=int, dest="img_size")
parser.add_argument("--img_channel", default=1, type=int, dest="img_channel")

parser.add_argument("--beta1", default=0.5, type=float, dest="beta1")
parser.add_argument("--gamma", default=0.3, type=float, dest="gamma")
parser.add_argument("--step_size", default=10, type=int, dest="step_size")

parser.add_argument("--real_label", default=1, type=int, dest="real_label")
parser.add_argument("--fake_label", default=0, type=int, dest="fake_label")

parser.add_argument('--name_data', type=str, default='axial', dest='name_data')
parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')

parser.add_argument('--dir_data', default='./datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--scope', default='', dest='scope')
parser.add_argument('--sim_mode', type=str, default='proposed', dest='sim_mode')
parser.add_argument('--pred_mode', type=str, default='pred', dest='pred_mode')

args = parser.parse_args()
    
PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()
    
    TRAINER = Train(ARGS)
    TRAINER.train()
    

if __name__ == "__main__":
     main()

