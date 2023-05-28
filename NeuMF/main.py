import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

from model import *
from config import parse_args, set_seed, print_args
from utils.core import Masking, CosineDecay, LinearDecay
from utils.data_loader import get_loader



args = parse_args()
print_args(args)
set_seed(seed=2022)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == "ml-1m":
    data_path = "./data/ml-1m/"
elif args.dataset == "citeulike":
    data_path = "./data/citeulike/"
elif args.dataset == "foursquare":
    data_path = "./data/foursquare/"


# UserID::MovieID::Rating::Timestamp (5-star scale)
train_loader = get_loader(data_path, args.train_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle, args.dataset)
val_loader = get_loader(data_path, args.val_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle, args.dataset)
test_loader = get_loader(data_path, args.test_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle, args.dataset)

# Getting the number of users and movies
if args.dataset == "ml-1m":
    num_users  = 6041
    num_movies = 3954
elif args.dataset == "citeulike":
    num_users  = 5220
    num_movies = 25189
elif args.dataset == "foursquare":
    num_users  = 15444
    num_movies = 28594

# Creating the architecture of the Neural Network
model = NeuMF(num_users, num_movies, args.emb_dim, args.layers)



if torch.cuda.is_available():
    model.cuda()

"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

#criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()#CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

mask=None

if args.sparse:

    if args.decay_schedule =='cosine':
        decay = CosineDecay(args.death_rate, len(train_loader)*(args.num_epochs))
    elif args.decay_schedule =='linear':
        decay = LinearDecay(args.death_rate, len(train_loader)*(args.num_epochs))
    else:
        assert False

    mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,redistribution_mode=args.redistribution, args=args)
    mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
else:
    print('Dense Training: mask is None.')

path = "model/ckpt-NeuMF-{}/".format(args.dataset)
os.makedirs(path, exist_ok=True)
name = path + "NeuMF-data-{}-bs-{}-lr-{}-{}-density-{}-T-{}".format(args.dataset, args.batch_size, args.lr, str(args.sparse), args.density, args.update_frequency)

best_epoch = 0
best_loss  = 9999.
best_hit = 0.0


def train():
    global best_loss, best_epoch, best_hit

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = 0
        model.train()
        for s, (x, n) in enumerate(train_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])

            pred, neg_pred = model(u, v, n)                
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            if mask is not None: mask.step()        #有mask就mask.step()
            else: optimizer.step()                  #没有mask就更新model参数

            if (s+1) % 500 == 0:
                print('epoch: '+str(epoch+1) + '|  batch: '+str(s+1)+'|  batch_loss: '+str(loss))
            # break
        print('epoch: '+str(epoch+1)+'|  loss: '+str(train_loss/(s+1)))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            val_hits = 0
            HR, NDCG = [], []
            with torch.no_grad():
                for s, (x, n) in enumerate(val_loader):
                    x = x.long().to(device)
                    n = n.long().to(device)
                    u = Variable(x[:,0])
                    v = Variable(x[:,1])
                    #r = Variable(x[:,2]).float()

                    pred, neg_pred = model(u, v, n)
                    loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                         + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
                    val_loss += loss.item()

                    pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)       #[512,101]
                    _, topk = torch.sort(pred, 1, descending=True)          #topk.shape: torch.Size([512, 101])
                    val_hit, val_ndcg = metric(topk, args.at_k)

            print('[val loss] : '+str(val_loss/(s+1))+' [val hit ratio] : '+str(val_hit)+' [val ndcg] : '+str(val_ndcg))
            if args.dataset != "citeulike":
                if best_loss > (val_loss/(s+1)):
                    best_loss = (val_loss/(s+1))
                    best_epoch= epoch+1

                    torch.save(model.state_dict(), name)
            else:
                if best_hit < (val_hit/(s+1)):
                    best_hit = (val_hit/(s+1))
                    best_epoch= epoch+1

                    torch.save(model.state_dict(), name)



def test():
    # Test
    print(name)
    model.load_state_dict(torch.load(name))
    model.eval()
    test_loss = 0
    test_hits = 0
    with torch.no_grad():
        for s, (x, n) in enumerate(test_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            #r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                 + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            test_loss += loss.item()

            # Hit Ratio
            pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
            _, topk = torch.sort(pred, 1, descending=True)
            test_hit, test_ndcg = metric(topk, args.at_k)
            
    print('[test loss] : '+str(test_loss/(s+1))+' [test hit ratio] : '+str(test_hit)+' [test ndcg] : '+str(test_ndcg))
    return test_hit, test_ndcg


def metric(topk_mat, tk):
    HR, NDCG = [], []
    for k in range(topk_mat.size(0)):
        if 0 in topk_mat[k, :tk]:
            hr_o = 1
            index = (topk_mat[k, :tk].tolist()).index(0)
            ndcg_o =  np.reciprocal(np.log2(index+2))
        else:
            hr_o = 0
            ndcg_o = 0
        HR.append(hr_o)
        NDCG.append(ndcg_o)

    hit = np.mean(HR)
    ndcg = np.mean(NDCG)
    return hit, ndcg





if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch

    r20, n20 = test()

    if mask is not None:
        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        print('The final percentage of the total fired weights is:', total_fired_weights)

    if mask is not None:
        print("DST: density:{}, update_frequency:{}, death_rate:{}, R:{}".format(args.density, args.update_frequency, args.death_rate, total_fired_weights))

    # print(" Finally: bs:{}, total_iteration:{}, lr_decay:{}, recall@20:{:.4f}, ndcg@20:{:.4f}".format(args.batch_size, batch_num*args.epochs, args.lr_decay, r20, n20))
    print(" Finally: bs:{}, total_iteration:{}, hit_ratio@20:{:.4f}, ndcg@20:{:.4f}".format(args.batch_size, len(train_loader)*args.num_epochs, r20, n20))
