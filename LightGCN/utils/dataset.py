from utils.data_util import create_link_prediction_dataset_test
import torch
import random
import numpy as np

class Train_Dataset(torch.utils.data.Dataset):
    # end2end
    def __init__(
            self,
            data=None,
            length=None,
            item_min=None,
            item_max=None,
            user_seq=None,
    ):
        super(Train_Dataset, self).__init__()
        self.item_min, self.item_max, self.user_seq = item_min, item_max, user_seq
        self.data = data
        self.length = length
        self.__update__()

    def sample_neg(self, x):
        while True:
            neg_id = random.randint(self.item_min, self.item_max)
            if neg_id not in x:
                return neg_id


    def __update__(self):
        self.user_idx = np.random.randint(0, self.item_min, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # user_idx = self.data.edge_index[0][idx]
        # item_idx = self.data.edge_index[1][idx]

        user_idx = self.user_idx[idx]
        item_idx = random.choice(self.user_seq[user_idx])
        # # # item_idx_neg
        item_idx_neg = self.sample_neg(self.user_seq[user_idx])
        return user_idx, item_idx , item_idx_neg


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, item_min, args):
        data = create_link_prediction_dataset_test(args.dataset)
        edge_index = data.edge_index.T.numpy()

        self.user_seq = {}
        j = 0
        for u in range(item_min):
            self.user_seq[u] = []
            while j < data.edge_index.shape[1] and edge_index[j][0] == u:
                self.user_seq[u].extend([(edge_index[j][1]-item_min).tolist()])
                j += 1

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, user):
        return user, self.user_seq[user]