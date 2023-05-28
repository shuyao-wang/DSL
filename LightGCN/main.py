import torch
import argparse
import numpy as np
import dgl
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from random import sample
from model import LightGCN
from utils.dataset import Train_Dataset, Test_Dataset
from utils.data_util import dataset_info, test_batcher
from utils.metrics import get_test, AverageMeter
from utils.core import Masking, CosineDecay, LinearDecay
from train import train, test, group_test_item, group_test_user





def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.random.seed(seed)

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="LightGCN + DST.")
    
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning Rate.')
    parser.add_argument('--lr_decay', nargs='+', type=int, default=[200, 600], help='Learning Rate.')
    parser.add_argument('--lr_decay_schedule', type=str, default='cosine', help='The decay schedule for the learning rate. Choose from: cosine, multisteplr')
    parser.add_argument('--eta_min', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dataset', type=str, default='yelp2018')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size(1024*N).')
    parser.add_argument('--epoch', type=int, default=1000, help='The inner loop max epoch')
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='regular')
    parser.add_argument("--topks", type=int, default=[20],help='use for test')
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument('--num_layer', type=int, default=3, help='light_gcn num layer.')
    parser.add_argument("--gpu", default=0, type=int, nargs='+', help="GPU id to use.")
    parser.add_argument("--seed", default=2022, type=int, help="seed")
    parser.add_argument('--part', type=str, default='join', help='user, item, user_item, join')
    parser.add_argument('--stop', type=str, default='nonstop', help='stop, nonstop')
    #DST settings
    parser.add_argument('--sparse', action='store_true',help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization. Choose from: lottery_ticket, uniform')
    parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--update_ratio', type=float, default=0.50, help='The pruning rate / growth rate.')
    parser.add_argument('--density', type=float, default=0.5, help='The density of the overall sparse network.')
    parser.add_argument('--update_interval', type=int, default=3000, metavar='N', help='how many iterations to train between parameter exploration')        #delta_T
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the update ratio. Default: cosine. Choose from: cosine, linear.')
    

    return parser.parse_args()


def main():
    args = parse_args()
    print_args(args)
    set_seed(seed=2022)


    item_min, item_max, data, length, user_seq, edge_index,\
    group_u_num, group_i_num, user_group, item_group = dataset_info(args.dataset)

    g = dgl.DGLGraph()
    g.add_nodes(item_max + 1)
    g.add_edges(edge_index[0], edge_index[1])
    g.add_edges(edge_index[1], edge_index[0])
    g = g.to("cuda:0")

    train_dataset = Train_Dataset(data=data, length=length, item_min=item_min, item_max=item_max, user_seq=user_seq, )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True,
                                               num_workers=2, 
                                               pin_memory=True)
    test_dataset = Test_Dataset(item_min, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              num_workers=2, 
                                              collate_fn=test_batcher(),)

    print("lightgcn!")
    model = LightGCN(g, args=args, edge_index=edge_index, item_max=item_max).cuda()
   
    optimizer = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    if args.lr_decay_schedule=='multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.5)
    
    if args.lr_decay_schedule=='cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.eta_min, last_epoch=-1)
    
    best_epoch = 0
    best_loss = {'loss': 0, 'emb': 0, 'reg': 0}
    
    args.start_epoch = 1
    best_result = {'precision': 0, 'recall': 0, 'ndcg': 0}

    total_time = 0


    mask=None

    if args.sparse:

        if args.decay_schedule =='cosine':
            decay = CosineDecay(args.update_ratio, len(train_loader)*(args.epoch))
        elif args.decay_schedule =='linear':
            decay = LinearDecay(args.update_ratio, len(train_loader)*(args.epoch))
        else:
            assert False

        mask = Masking(optimizer, update_ratio=args.update_ratio, death_mode=args.death, update_ratio_decay=decay, growth_mode=args.growth,redistribution_mode=args.redistribution, args=args)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
    else:
        print('Dense Training: mask is None.')



    for epoch in range(args.start_epoch, args.epoch + 1):
        t0 = time.time()
        train_dataset.__update__()

        loss, re_loss, reg_loss = train(train_loader, model, optimizer, args, mask)
        result = test(test_loader, model, item_min, user_seq, args)
        lr_scheduler.step()
        if result['recall'] > best_result['recall']:
            best_result = result
            best_epoch = epoch
            best_loss['loss'] = loss
            best_loss['emb'] = re_loss
            best_loss['reg'] = reg_loss




        epoch_time = (time.time() - t0) / 60
        total_time += epoch_time
        print('Epoch:[{}/{}] loss:[{:.4f}], emb:[{:.4f}], reg:[{:.4f}] | recall:[{:.4f}], ndcg:[{:.4f}], prec:[{:.4f}] | Update at epoch:[{}] recall:[{:.4f}], ndcg:[{:.4f}], lr:[{:.4f}] | time:[{:.2f}min] Total:[{:.2f}min]'
                .format(epoch, 
                        args.epoch, 
                        loss, 
                        re_loss, 
                        reg_loss, 
                        result['recall'], 
                        result['ndcg'],
                        result['precision'], 
                        best_epoch,
                        best_result['recall'], 
                        best_result['ndcg'],
                        optimizer.param_groups[0]['lr'],
                        epoch_time,
                        total_time
                        ))
        
    

    if mask is not None:
        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        print('The final percentage of the total fired weights is:', total_fired_weights)
    
    if mask is not None:
        print("DST: density:{}, update_interval:{}, update_ratio:{}, R:{}".format(args.density, args.update_interval, args.update_ratio, total_fired_weights))

    print(" Finally: bs:{}, total_iteration:{}, lr_decay:{}, recall:{:.4f}, ndcg:{:.4f}".format(args.batch_size, len(train_loader)*args.epoch, args.lr_decay, best_result['recall'], best_result['ndcg']))


if __name__ == '__main__':
    main()