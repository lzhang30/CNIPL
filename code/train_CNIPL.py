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
if round(float(torch.version.cuda)) > 10:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset_semi import (BaseDataSets, RandomGenerator,RandomGenerator_Strong_Weak,RandomGenerator_CCM,
                                      TwoStreamBatchSampler)
from networks.discriminator import FCDiscriminator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume,test_single_volume_val,test_co_volume_val,test_single_volume_co_save

from utils.mixup import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ours_ws', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--fold', type=str,
                    default='fold1', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--early_stop', type=int, default=10000,
                    help='early_stop')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--check', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--labeled_ratio', type=int,  default=8,
                    help='output channel of network')
# costs


parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')

parser.add_argument('--lamda', type=float,
                    default=1, help='consistency')

parser.add_argument('--choice', type=str,
                    default='ccm', help='mix type')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def custom_operation(prob_A,prob_B):
    
    ext = torch.nn.functional.pad(torch.clone(prob_B), (1,1,1,1,0,0,0,0))
    left = ext[:,:,:-2, 1:-1]
    right = ext[:,:,2:,1:-1]
    up = ext[:,:,1:-1,:-2]
    down = ext[:,:,1:-1, 2:]
    left_up = ext[:, :, :-2, :-2]
    left_down = ext[:, :, :-2, 2:]
    right_up = ext[:, :, 2:, :-2]
    right_down = ext[:, :, 2:, 2:]
    d = torch.stack((left, right, up, down, left_up, left_down, right_up, right_down, prob_B)).to(prob_A.device)
    arr,neigbor_idx=torch.max(d,0) 
    beta = torch.exp(torch.tensor(-1/2))
    prob = prob_A + beta*arr - (prob_A*arr*beta)
    #output = prob.max(1)[1]
    
    return prob


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    model1_ema = create_model(ema=True)
    model2_ema = create_model(ema=True)
    

    
    model1 = kaiming_normal_init_weight(model1)
    model2 = xavier_normal_init_weight(model2)
    
    model1_ema = xavier_normal_init_weight(model1_ema)
    model2_ema = kaiming_normal_init_weight(model2_ema)
    #teacher = create_model(ema=True)

    db_train_labeled = BaseDataSets(base_dir=args.root_path, num=8, ratio = args.labeled_ratio, labeled_type="labeled", fold=args.fold, split="train", sup_type=args.sup_type, transform=transforms.Compose([
        RandomGenerator_Strong_Weak(args.patch_size)
    ]))
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="unlabeled", fold=args.fold, split="train", sup_type=args.sup_type, transform=transforms.Compose([
        RandomGenerator_Strong_Weak(args.patch_size)]))

    trainloader_labeled = DataLoader(db_train_labeled, batch_size=args.batch_size//2, shuffle=True,
                                     num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.batch_size//2, shuffle=True,
                                       num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="test", )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    
    
    best_iter = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    model1.train()
    model2.train()
    
    
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,\
                            momentum=0.9, weight_decay=0.0001)
                           

    ce_loss = CrossEntropyLoss(ignore_index=args.num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader_labeled)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_labeled) + 1

    # randomly generate one aug for each iteration



    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i, data in enumerate(zip(cycle(trainloader_labeled), trainloader_unlabeled)):
            sampled_batch_labeled, sampled_batch_unlabeled = data[0], data[1]
            
            labeled_volume_batch_weak,labeled_volume_batch_strong, label_batch = \
                sampled_batch_labeled['image_w'].cuda(), sampled_batch_labeled['image_s'].cuda(), sampled_batch_labeled['label'].cuda()
            #print(labeled_volume_batch_raw.shape,labeled_volume_batch_weak.shape)
            

            
            
            #torch.save(label_batch, 'label_batch.pth')
            #break
            #labeled_volume_batch, label_batch = labeled_volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch_strong,unlabeled_volume_batch_weak = \
                sampled_batch_unlabeled['image_s'].cuda(),sampled_batch_unlabeled['image_w'].cuda()
            #print("Labeled slices: ", sampled_batch_labeled["idx"])
            #print("Unlabeled slices: ", sampled_batch_unlabeled["idx"])

            #sup loss
            
            output_sup1_weak = model1(labeled_volume_batch_weak)
            output_sup2_weak = model2(labeled_volume_batch_weak)
            
            
            
            output_sup1_strong = model1(labeled_volume_batch_strong)
            output_sup2_strong = model2(labeled_volume_batch_strong)
            

            
            loss_ce = 0.25*ce_loss(output_sup1_weak, label_batch[:].long())+\
                        0.25 * ce_loss(output_sup2_weak, label_batch[:].long())+\
                            0.25*ce_loss(output_sup1_strong, label_batch[:].long())+\
                                0.25*ce_loss(output_sup2_strong, label_batch[:].long())
       
            
            #one_hot_labels = F.one_hot(label_batch.long(), num_classes).float()
            #one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)  # [b, c, w, h]


            #print(choice)
            
            
            volume_combined_weak= torch.cat((labeled_volume_batch_weak, unlabeled_volume_batch_weak), dim=0)
            volume_combined_strong = torch.cat((labeled_volume_batch_strong, unlabeled_volume_batch_strong), dim=0)
            with torch.no_grad():
                model1.eval()
                model2.eval()
                outputs1_unlabeled_weak = model1_ema(volume_combined_weak)
                weak_label1 = torch.softmax(outputs1_unlabeled_weak, dim=1).detach()
            

                #outputs1_unlabeled_strong = model1_ema(volume_combined_strong)
                #strong_label1 = torch.softmax(outputs1_unlabeled_strong, dim=1).detach()
                
                outputs2_unlabeled_weak = model2_ema(volume_combined_weak)
                weak_label2 = torch.softmax(outputs2_unlabeled_weak, dim=1).detach()
                
                #outputs2_unlabeled_strong = model2_ema(volume_combined_strong)
                #strong_label2 = torch.softmax(outputs2_unlabeled_strong, dim=1).detach()
                model1.train()
                model2.train()
                
                beta = random.random()
                beta1 = random.random()
                beta2 = random.random()
                
                
                
                #mix_label = beta * custom_operation(weak_label1,weak_label2)+\
                #    (1-beta) * custom_operation(weak_label2,weak_label1)
                weak_label = beta1* custom_operation(weak_label2,weak_label1)+\
                    (1-beta1) * custom_operation(weak_label1,weak_label2)
                #weak_label2 = beta2 * custom_operation(weak_label2,strong_label2)+\
                #    (1-beta2) * custom_operation(strong_label2,weak_label2)
                  
                    
                      
            shape = list(volume_combined_weak.shape)
            shape[1] = 1
            #print(shape)
            MixMask = generate_cutmix_mask(shape=shape).cuda().float()
            rand_index = torch.randperm(volume_combined_weak.size()[0]).cuda()
            pseudo_lab = MixMask * weak_label + (1 - MixMask) * weak_label[rand_index]
            #pseudo_lab2 = MixMask * weak_label2 + (1 - MixMask) * weak_label2[rand_index]
            #pseudo_mix = MixMask * mix_label + (1 - MixMask) * mix_label[rand_index]
            
            pseudo_lab = torch.argmax(pseudo_lab, dim=1)
            #pseudo_lab2 = torch.argmax(pseudo_lab2, dim=1)
            #pseudo_lab_mix = torch.argmax(pseudo_mix, dim=1)

            volume_combined_strong = MixMask * volume_combined_strong + (1 - MixMask) * volume_combined_strong[rand_index]
            
            volume_combined_weak = MixMask * volume_combined_weak + (1 - MixMask) * volume_combined_weak[rand_index]
            
            #volume_combined_mix = MixMask * volume_combined_weak + (1 - MixMask) * volume_combined_weak[rand_index]
            
            output_weak1 = model1(volume_combined_weak)
            output_strong1 = model1(volume_combined_strong)
          
            output_weak2 = model2(volume_combined_weak)
            output_strong2 = model2(volume_combined_strong)
           
            loss_unsup = 0.25*F.cross_entropy(output_weak1, pseudo_lab )+\
                        0.25*F.cross_entropy(output_strong1, pseudo_lab)+\
                                0.25*F.cross_entropy(output_weak2, pseudo_lab)+\
                                    0.25*F.cross_entropy(output_strong2, pseudo_lab)
            
            
                         
            loss = loss_ce + args.lamda * loss_unsup
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            update_ema_variables(model1, model1_ema, args.ema_decay, iter_num)
            update_ema_variables(model2, model2_ema, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_unsup', loss_unsup, iter_num)


                
            if iter_num > 0 and iter_num % 1000 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f, loss_unsup: %f' %
                    (iter_num, loss.item(), loss_ce.item(), loss_unsup.item()))


            if iter_num >= 0 and iter_num % args.check == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i, prediction = test_single_volume_val(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    
                image = sampled_batch["image"][0, 0:1, :, :]
                #print(sampled_batch["image"][0].max())
                writer.add_image('train/val_Image_val', image/image.max(), iter_num)
                #print(sampled_batch["label"].shape)
                
                
                writer.add_image('train/val_Image_label', sampled_batch["label"][0,0:1 ,...].long()/4, iter_num)
                
                writer.add_image('train/val_model_Prediction1',
                                 prediction[0:1, ...]/4, iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                if performance > best_performance1:           
                    best_performance1 = performance
                    best_iter = iter_num
                    torch.save(model1.state_dict(), os.path.join(
                        snapshot_path, 'model_best1.pth'))
                    logging.info('best model1 found, best_iter: %d' % best_iter)
                    
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)


                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f model 1' % (iter_num, performance, mean_hd95))
                model1.train()
                
                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i,prediction = test_single_volume_val(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                
                
                
                writer.add_image('train/val_model_Prediction2',
                                 prediction[0:1, ...]/4, iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                if performance > best_performance2:           
                    best_performance2 = performance
                    best_iter = iter_num
                    torch.save(model2.state_dict(), os.path.join(
                        snapshot_path, 'model_best2.pth'))
                    logging.info('best model2 found, best_iter: %d' % best_iter)
                
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)


                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f model 2' % (iter_num, performance, mean_hd95))
                model2.train()



            if iter_num >= max_iterations or iter_num- best_iter > args.early_stop: 
                
                break
        if iter_num >= max_iterations or iter_num- best_iter > args.early_stop:
            iterator.close()
            break
    model1.load_state_dict(torch.load(os.path.join(
                    snapshot_path, 'model_best1.pth')))
    
    model2.load_state_dict(torch.load(os.path.join(
        snapshot_path, 'model_best2.pth')))
    model1.eval()
    model2.eval()

    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(valloader):
        metric_i, prediction = test_co_volume_val(
            sampled_batch["image"], sampled_batch["label"], model1,model2, classes=num_classes)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_val)
    for class_i in range(num_classes-1):
        writer.add_scalar('info/final_val_{}_dice'.format(class_i+1),
                            metric_list[class_i, 0], iter_num)
        writer.add_scalar('info/final_val_{}_hd95'.format(class_i+1),
                            metric_list[class_i, 1], iter_num)
    
    
    metric_list = 0.0
    test_ids = sorted(os.listdir(os.path.join(args.root_path , "test_volumes")))
    for case in test_ids:
        metric_i = test_single_volume_co_save(case, net1 = model1, net2 = model2, test_save_path=os.path.join(snapshot_path,'result'), FLAGS=args, batch_size=12)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(test_ids)
    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
                'val : model_mean_dice : %f model_mean_hd95 : %f' % (performance, mean_hd95))
    '''
    
    metric_list = 0.0        
    #print (f'test save path {os.path.join(snapshot_path,'result')}')
    for case in test_ids:
        metric_i = test_single_volume_co_save(case, net1 = model1, net2 = model2, test_save_path=os.path.join(snapshot_path,'result'), FLAGS=args, batch_size=12)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(test_ids)
    performance = np.mean(metric_list, axis=0)[0]

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
                'test : model_mean_dice : %f model_mean_hd95 : %f' % (performance, mean_hd95))
    
    '''
    writer.close()
    return "Training Finished!"


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

    snapshot_path = "../model_WSS/{}/{}/{}".format(
        args.exp,  args.labeled_ratio,args.fold)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
