import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensorboardX import SummaryWriter
from scipy import sparse
import models
from util.data import *
from util.metric import NDCG_binary_at_k_batch, Recall_at_k_batch
from util.core import Masking, CosineDecay, LinearDecay



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
    parser.add_argument('--data', type=str, default='ml-1m',
                        help="[ml-1m],[citeulike],[foursquare]")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    
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

    return parser.parse_args()

###############################################################################
# args
###############################################################################
args = parse_args()
print_args(args)
set_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
if args.data == 'ml-1m':
    data_path = './data/ml-1m'
elif args.data =='citeulike':
    data_path = './data/citeulike'
elif args.data =='foursquare':
    data_path = './data/foursquare'
else:
    assert False

loader = DataLoader(data_path)

n_items = loader.load_n_items()
train_data = loader.load_data('train')
vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')

N = train_data.shape[0]
idxlist = list(range(N))

###############################################################################
# Build the model
###############################################################################

p_dims = [200, 600, n_items]
model = models.MultiVAE(p_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
criterion = models.loss_function
batch_num = (N // args.batch_size) + 1


###############################################################################
# DSL Init
###############################################################################
mask=None

if args.sparse:

    if args.decay_schedule =='cosine':
        decay = CosineDecay(args.death_rate, batch_num*(args.epochs))
    elif args.decay_schedule =='linear':
        decay = LinearDecay(args.death_rate, batch_num*(args.epochs))
    else:
        assert False

    mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,redistribution_mode=args.redistribution, args=args)
    mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
else:
    print('Dense Training: mask is None.')


###############################################################################
# Training code
###############################################################################

# TensorboardX Writer

writer = SummaryWriter()

def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i : row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train():
    # Turn on training mode
    # pdb.set_trace()
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        
        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        
        if mask is not None: mask.step()        #有mask就mask.step()
        else: optimizer.step()                  #没有mask就更新model参数
        # optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            
            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, args.batch_size)) + batch_idx
            writer.add_scalars('data/loss', {'train': train_loss / args.log_interval}, n_iter)

            start_time = time.time()
            train_loss = 0.0



def evaluate(data_tr, data_te):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    # pdb.set_trace()
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n20_list = []
    r20_list = []
    n50_list = []
    r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n20 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            n50 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 50)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n20_list.append(n20)
            r20_list.append(r20)
            n50_list.append(n50)
            r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n20_list = np.concatenate(n20_list)
    r20_list = np.concatenate(r20_list)
    n50_list = np.concatenate(n50_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n20_list), np.mean(n50_list), np.mean(r20_list), np.mean(r50_list)


best_n20 = -np.inf
update_count = 0
best_epoch = 0
# At any point you can hit Ctrl + C to break out of training early.
try:

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # pdb.set_trace()
        train()
        # pdb.set_trace()
        val_loss, n20, n50, r20, r50 = evaluate(vad_data_tr, vad_data_te)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
                'n20 {:5.3f} | n50 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, time.time() - epoch_start_time, val_loss,
                    n20, n50, r20, r50))
        print('-' * 89)

        n_iter = epoch * len(range(0, N, args.batch_size))
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalar('data/n100', n20, n_iter)
        writer.add_scalar('data/n100', n50, n_iter)
        writer.add_scalar('data/r20', r20, n_iter)
        writer.add_scalar('data/r50', r50, n_iter)

        # Save the model if the n100 is the best we've seen so far.
        if n20 > best_n20:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_n20 = n20
            best_epoch = epoch

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.

test_loss, n20, n50, r20, r50 = evaluate(test_data_tr, test_data_te)
print('=' * 89)
# print('| End of training | test loss {:4.2f} | n20 {:4.2f} | n50 {:4.2f} | r20 {:4.2f} | '
#         'r50 {:4.2f}'.format(test_loss, n20, n50, r20, r50))
print('| End of training | test loss {:4f} | n20 {:4f} | n50 {:4f} | r20 {:4f} | '
        'r50 {:4f}'.format(test_loss, n20, n50, r20, r50))
print('=' * 89)

if mask is not None:
    layer_fired_weights, total_fired_weights = mask.fired_masks_update()
    print('The final percentage of the total fired weights is:', total_fired_weights)

if mask is not None:
    print("DST: density:{}, update_frequency:{}, death_rate:{}, R:{}".format(args.density, args.update_frequency, args.death_rate, total_fired_weights))

# print(" Finally: bs:{}, total_iteration:{}, lr_decay:{}, recall@20:{:.4f}, ndcg@20:{:.4f}".format(args.batch_size, batch_num*args.epochs, args.lr_decay, r20, n20))
print(" Finally: bs:{}, best_epoch:{}, recall@20:{:.4f}, ndcg@20:{:.4f}".format(args.batch_size, best_epoch, r20, n20))