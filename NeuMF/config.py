import argparse
import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--mode', type=str, default="train", help='train / test')
    parser.add_argument('--dataset', type=str, default="ml-1m", help="[ml-1m],[citeulike],[foursquare]")
    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--val-step', type=int, default=5)

    parser.add_argument('--test-epoch', type=int, default=1)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--neg-cnt', type=int, default=100)
    parser.add_argument('--at_k', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--layers', default=[32,32,16,8])


    parser.add_argument('--train-path', type=str, default='train_score.pkl')
    parser.add_argument('--val-path', type=str, default='val_score.pkl')
    parser.add_argument('--test-path', type=str, default='test_score.pkl')
    parser.add_argument('--neg-path', type=str, default='neg_score.npy')


    #DST settings
    parser.add_argument('--sparse', action='store_true',help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    #parser.add_argument('--fix', default='True', help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization. Choose from: lottery_ticket, uniform')
    parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.5, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between parameter exploration')        #delta_T
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')


    args = parser.parse_args()

    return args
