import argparse
import logging
import os
import random
import shutil
import sys
import time
from itertools import cycle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume,test_single_volume_co_save,test_single_volume_single_save,test_single_volume_single_save_cct

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/cps', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=int,
                    default=1, help='cross validation')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--cross_val', type=bool,
                    default=True, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_ratio', type=int, default=8,
                    help='1/labeled_ratio data is provided mask')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)




def test_dan(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    if 'dan' in args.exp:
        args.model = 'unet'
    model = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)

    
    try:
        print(snapshot_path)
        model.load_state_dict(torch.load(os.path.join(snapshot_path,
                                        'unet_best_model.pth'.format(args.model))))
    except:
        try:
            model.load_state_dict(torch.load(os.path.join(snapshot_path,
                                        'best_model.pth'.format(args.model))))
            
        except:
            model.load_state_dict(torch.load(os.path.join(snapshot_path,
                                            'model_best1.pth'.format(args.model))))

    
    test_ids = sorted(os.listdir(args.root_path + "/test_volumes"))    
    model.eval()

    metric_list = 0.0        
    #print (f'test save path {os.path.join(snapshot_path,'result')}')
    for case in test_ids:
        metric_i = test_single_volume_single_save(case, model, test_save_path=os.path.join(snapshot_path,'result'), FLAGS=args, batch_size=12)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(test_ids)
    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
                'test : model_mean_dice : %f model_mean_hd95 : %f' % (performance, mean_hd95))
    writer.close()
    return metric_list



def test_cct(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    args.model = 'unet_cct'
    model = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)

    save_latest = os.path.join(
    snapshot_path, '{}_latest_model.pth'.format(args.model))
    #torch.save(model1.state_dict(), save_latest)
    save_latest = os.path.join(
        snapshot_path, '{}_latest_model.pth'.format(args.model))
    #torch.save(model2.state_dict(), save_latest)
    try:
        model.load_state_dict(torch.load(os.path.join(snapshot_path,
                                        'unet_best_model.pth'.format(args.model))))
    except:
        try:
            model.load_state_dict(torch.load(os.path.join(snapshot_path,
                                            'best_model.pth'.format(args.model))))
        except:
            model.load_state_dict(torch.load(os.path.join(snapshot_path,'scribble',
                                            'model_best.pth'.format(args.model))))

    
    test_ids = sorted(os.listdir(args.root_path + "/test_volumes"))    
    model.eval()

    metric_list = 0.0        
    #print (f'test save path {os.path.join(snapshot_path,'result')}')
    for case in test_ids:
        metric_i = test_single_volume_single_save_cct(case, model, test_save_path=os.path.join(snapshot_path,'result'), FLAGS=args, batch_size=12)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(test_ids)
    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
                'test : model_mean_dice : %f model_mean_hd95 : %f' % (performance, mean_hd95))
    writer.close()
    return metric_list

def test(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    model1 = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)
    model2 = net_factory(net_type=args.model, in_chns=1,
                         class_num=num_classes)


    model1.load_state_dict(torch.load(os.path.join(snapshot_path,
                                        'model_best1.pth'.format(args.model))))
    model2.load_state_dict(torch.load(os.path.join(snapshot_path,
                                            'model_best2.pth'.format(args.model))))
    
    test_ids = sorted(os.listdir(args.root_path + "/test_volumes"))
    model1.eval()
    model2.eval()
    metric_list = 0.0        
    #print (f'test save path {os.path.join(snapshot_path,'result')}')
    for case in test_ids:
        metric_i = test_single_volume_co_save(case, net1 = model1, net2 = model2, test_save_path=os.path.join(snapshot_path,'result'), FLAGS=args, batch_size=12)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(test_ids)
    #print(f'metric_list: {metric_list}')
    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
                'test : model_mean_dice : %f model_mean_hd95 : %f' % (performance, mean_hd95))
    writer.close()
    return metric_list


import  csv

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #val = []

    
    title = ['dice1', 'std1', 'hd951', 'std1', 'dice2', 'std2', 'hd952', 'std2', 'dice3', 'std3', 'hd953', 'std3', 'avgdice','avghd95']
    # finished 'ACDC_train_d' 'ACDC_train_f','ACDC_train_cps','ACDC_train_dan','ACDC_train_dfcps',\
                     # 'ACDC_train_dmpl','ACDC_train_s4mc','ACDC_train_unimatch_test'
    #,'ACDC_train_a','ACDC_train_b','ACDC_train_c','ACDC_train_a_t','ACDC_train_b','ACDC_train_c','ACDC_train_e',


    print('test start')
    for args.exp in  ['ACDC_dmpl_ours_wsw']:
        for args.labeled_ratio in [8,16]:
            val = []
            for args.fold in [1,2,3,4,5]:

                print(f'start {args.exp} fold {args.fold}')

                snapshot_path = "../model_WSS/{}/{}/fold{}".format(
                    args.exp, args.labeled_ratio, args.fold)
                
                
                #print(snapshot_path)
                        #logging.info(str(args))
                        
                if 'dan' in  args.exp or 'uamt' in args.exp or 's4mc' in args.exp:
                    
                    args.model = 'unet'
                    metric_list = test_dan(args, snapshot_path)
                    
                elif 'unimatch' in args.exp or 'dmpl' in args.exp:
                    args.model = 'unet_cct'
                    metric_list = test_cct(args, snapshot_path)
                else:
                    args.model = 'unet'
                    try:
                        
                        metric_list = test(args, snapshot_path)
                    except:
                        snapshot_path = snapshot_path = "../model_WSS/{}/{}/fold{}/scribble".format(args.exp, args.labeled_ratio, args.fold)
                        metric_list = test(args, snapshot_path)
                val.append(metric_list)
                logging.info(metric_list)
            
            data = np.array(val)
            #print(data,data.shape)
            mean_per_class = np.mean(data, axis=0)
            std_per_class = np.std(data, axis=0)
            
            dice = list(mean_per_class[:, 0])
            hd95 = list(mean_per_class[:, 1])
            asd = list(mean_per_class[:, 2])
            dice_std = list(std_per_class[:, 0])
            hd95_std = list(std_per_class[:, 1])
            asd_std = list(std_per_class[:, 2])
            #print(f'dice: {dice}')
            #print(f'hd95: {hd95}')
            writter_row = [
                args.exp,
                f'{dice[0]*100:.1f}({dice_std[0]*100:.1f})', 
                f'{hd95[0]:.1f}({hd95_std[0]:.1f})', 
                f'{asd[0]:.1f}({asd_std[0]:.1f})', 
                f'{dice[1]*100:.1f}({dice_std[1]*100:.1f})', 
                f'{hd95[1]:.1f}({hd95_std[1]:.1f})', 
                f'{asd[1]:.1f}({asd_std[1]:.1f})', 
                f'{dice[2]*100:.1f}({dice_std[2]*100:.1f})', 
                f'{hd95[2]:.1f}({hd95_std[2]:.1f})', 
                f'{asd[2]:.1f}({asd_std[2]:.1f})',
                f'{np.mean(dice)*100:.1f}({np.mean(dice_std)*100:.1f})', 
                f'{np.mean(hd95):.2f}({np.mean(hd95_std):.1f})', 
                f'{np.mean(asd):.1f}({np.mean(asd_std):.1f})'
            ]
            
            save_csv = f'output_{args.labeled_ratio}.csv'

            with open(save_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                csvwriter.writerow(writter_row)
            print(writter_row)
            #logging.info(f'mean_per_class: {mean_per_class}')
            #logging.info(f'std_per_class: {std_per_class}')

            writer_row = []
            
            writer_row  = writer_row + []

            #logging.info(f'mean_per_class: {", ".join(map(str, mean_per_class))}')

            # Calculate the overall mean and standard deviation

            #overall_mean = list(np.mean(data, axis=(0, 1)))
            #overall_std = np.std(data, axis=(0, 1))()
            #logging.info(f'overall_mean: {", ".join(map(str, overall_mean))}')
            #logging.info(f'overall_std: {", ".join(map(str, overall_std))}')


    
    '''    
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    '''
    
