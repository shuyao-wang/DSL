import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class GMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(GMF, self).__init__()
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)

    def forward(self, u, v):
        u = self.u_emb(u)
        v = self.v_emb(v)
        return torch.mul(u, v)

class NeuMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, layers):
        super(NeuMF, self).__init__()

        self.gmf = GMF(num_user, num_item, emb_dim)
        self.u_emb = nn.Embedding(num_user, emb_dim)
        self.v_emb = nn.Embedding(num_item, emb_dim)
        linears = []
        for (in_d, out_d) in zip(layers[:-1], layers[1:]):
            linears.append(nn.Linear(in_d, out_d))
            linears.append(nn.ReLU())
            linears.append(nn.Dropout(p=0.2))
        self.linear = nn.Sequential(*linears)
        self.predict= nn.Linear(emb_dim+layers[-1], 1)

    def forward(self, u, v, n):
        # GMF
        
        gmf = self.gmf(u,v)
        gmf_n=self.gmf(u.unsqueeze(1).expand_as(n),n).view(-1,gmf.size(-1))

        # MLP
        u = self.u_emb(u)
        v = self.v_emb(v)
        n = self.v_emb(n)
        x = torch.cat((u, v), 1)
        x_n=torch.cat((u.unsqueeze(1).expand_as(n), n), 2)
        x_n=x_n.view(-1,x_n.size(-1))

        h = self.linear(x)
        h_n=self.linear(x_n)

        # Fusion
        pred = self.predict(torch.cat((gmf,h), 1)).view(-1)
        pred_n=self.predict(torch.cat((gmf_n,h_n), 1)).view(-1)

        return pred, pred_n
