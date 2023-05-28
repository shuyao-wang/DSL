import os
from collections import namedtuple
import numpy as np
import torch
import pandas as pd
import heapq

Data = namedtuple("Data", ["edge_index"])

def test_batcher():
    def batcher_dev(batch):
        users, items = zip(*batch)
        items = list(items)
        users = list(users)
        return users, items
    return batcher_dev

def train_batcher_negsample():
    def batcher_dev(batch):
        users, items, items_neg = zip(*batch)
        items = list(items)
        users = list(users)
        items_neg = list(items_neg)
        return users, items, items_neg
    return batcher_dev

def train_batcher():
    def batcher_dev(batch):
        users, items = zip(*batch)
        items = torch.tensor(items, dtype=torch.long)
        users = torch.tensor(users, dtype=torch.long)
        return users, items
    return batcher_dev


def dataset_info(dataset):  #return the dataset information and hot hash
    
    data = create_link_prediction_dataset_train(dataset)
    item_min = data.edge_index[1].min().item() # item id min: 36188
    item_max = data.edge_index[1].max().item() # item id max: 69715
    edge_index = data.edge_index.T.numpy()
    length = data.edge_index.shape[1]          # interactions: 1237259

    user_seq = {}  # user to many items
    j = 0
    for u in range(item_min):
        user_seq[u] = []
        while j < length and edge_index[j][0] == u:
            user_seq[u].extend([edge_index[j][1].tolist()])
            j += 1
    
    n_user_group = 10
    u_degree = np.array([len(user_seq[i]) for i in range(len(user_seq))])
    u_degree_sort = np.argsort(u_degree)
    u_degree_cumsum = u_degree.copy()
    cum_sum = 0
    for x in u_degree_sort:
        cum_sum += u_degree_cumsum[x]
        u_degree_cumsum[x] = cum_sum

    
    split_idx = np.linspace(0, length, n_user_group + 1)
    user_group = np.searchsorted(split_idx[1:-1], u_degree_cumsum)
    group_u_num = []
    for i in range(n_user_group):
        group_i = u_degree[user_group == i]  #get the degree
        group_u_num.append(group_i.sum())   # each group interaction num

    # item
    n_item_group = 10
    i_degree = np.bincount(np.concatenate([user_seq[i] for i in range(len(user_seq))]).astype(np.int32))
    true_i_degree = i_degree[item_min:]
    i_degree_sort = np.argsort(true_i_degree)
    i_degree_cumsum = true_i_degree.copy()
    cum_sum = 0
    for x in i_degree_sort:
        cum_sum += i_degree_cumsum[x]
        i_degree_cumsum[x] = cum_sum

    split_idx = np.linspace(0, length, n_item_group + 1)
    item_group = np.searchsorted(split_idx[1:-1], i_degree_cumsum)
    group_i_num = []
    for i in range(n_item_group):
        group_i = true_i_degree[item_group == i]
        group_i_num.append(group_i.sum())
        # print("Size of group %d:" % i, group_i.size)
        # print("Sum degree of group %d:" % i, group_i.sum())
        # print("Min degree of group %d:" % i, group_i.min())
        # print("Max degree of group %d:" % i, group_i.max())
    return item_min, item_max, data, length, user_seq, data.edge_index, \
            group_u_num, group_i_num, user_group, item_group  # hot_idx is node id - item_min ,all item id,  all-item_min

def read_data(path):
    data = pd.read_csv(path, sep=' ', header=None, names=["user", "item"])
    edge_list = data.values.tolist()
    data = Data(edge_index=torch.LongTensor(edge_list).t())
    return data

def create_link_prediction_dataset(name):
    link_list_path = os.path.join("./data/", name + ".full")
    return read_data(link_list_path)

def create_link_prediction_dataset_train(name):
    link_list_path = os.path.join("./data/", name + ".train")
    return read_data(link_list_path)

def create_link_prediction_dataset_test(name):
    link_list_path = os.path.join("./data/", name + ".test")
    return read_data(link_list_path)

if __name__ == "__main__":
    a = create_link_prediction_dataset("ifashion")
    print(a)
