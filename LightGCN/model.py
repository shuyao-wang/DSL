import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import copy


gcn_msg = fn.src_mul_edge('h', 'adj', 'm')  #e is norm,
gcn_reduce = fn.sum(msg='m', out='h')  # source add, edge update!!


class LightGCN(nn.Module):
    def __init__(self, g, args, edge_index, item_max):
        super(LightGCN, self).__init__()
        self.reg_weight = args.reg_weight
        self.num_layer = args.num_layer
        self.dataset = args.dataset

        self.g = g
        self.mask = None

        self.gpu = args.gpu
        self.embedding = nn.Embedding(num_embeddings=item_max + 1, embedding_dim=args.embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.norm()

        self.neg = None
        self.pos = None
        self.emb_list = None

    def norm(self):  #from mask get adj edata
        # self.g = dgl.transform.remove_self_loop(self.g)  #remove
        # self.g = dgl.add_self_loop(self.g)  # add self loop
        self.g.edata["mask"] = torch.ones(self.g.number_of_edges()).unsqueeze(dim=1).to(torch.device(self.gpu))
        self.g.update_all(fn.copy_e('mask', 'm'), fn.sum('m', 'd'))
        self.g.ndata["d"] = self.g.ndata["d"] ** -0.5
        self.g.ndata["d"][self.g.ndata["d"] == float("inf")] = 0
        self.g.apply_edges(fn.u_mul_e('d', 'mask', 'adj'))
        self.g.apply_edges(fn.e_mul_v('adj', 'd', 'adj'))

    def get_emb(self):
        #pdb.set_trace()
        embeddings_list = [self.embedding.weight]
        self.g.ndata['h'] = embeddings_list[-1]

        for layer in range(self.num_layer):
            self.g.update_all(gcn_msg, gcn_reduce)
            embeddings_list.append(self.g.ndata['h'])

        self.emb_list = embeddings_list
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings


    def forward(self, users, pos, neg):
        #pdb.set_trace()
        all_emb = self.get_emb()

        users_emb = all_emb[users]
        pos_emb = all_emb[pos]
        neg_emb = all_emb[neg]

        users_emb_ego = self.embedding(users)
        pos_emb_ego = self.embedding(pos)
        neg_emb_ego = self.embedding(neg)
        #pdb.set_trace()
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                                pos_emb_ego.norm(2).pow(2) +
                                neg_emb_ego.norm(2).pow(2)) / float(len(users))
        loss_emb = self.create_bpr_loss(users_emb, pos_emb, neg_emb)

        reg_loss = reg_loss * self.reg_weight
        loss = loss_emb + reg_loss
        return loss, loss_emb,  reg_loss
        
    def create_bpr_loss(self, users_emb, pos_emb, neg_emb):
        #pdb.set_trace()
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        self.neg = neg_scores
        self.pos = pos_scores
        loss_emb = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss_emb