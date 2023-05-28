import numpy as np
import pandas as pd
import random

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

num_users = len(users)
num_movies= len(movies)

train_lst = []
val_lst   = []
test_lst  = []

for uid in range(num_users):
    watches = ratings.loc[ratings[0] == uid]
    
    train_lst.append(watches.iloc[:int(len(watches)*0.7)])
    val_lst.append(watches.iloc[int(len(watches)*0.7):int(len(watches)*0.8)])
    test_lst.append(watches.iloc[int(len(watches)*0.8):])

train = pd.concat(train_lst)
val   = pd.concat(val_lst)
test  = pd.concat(test_lst)
train.to_pickle('./data/ml/train.pkl')
val.to_pickle('./data/ml/val.pkl')
test.to_pickle('./data/ml/test.pkl')


train_lst = []
val_lst   = []
test_lst  = []
neg_lst  = []

for uid in range(1, num_users+1):
    watches = ratings.loc[ratings[0] == uid]
    
    watched = watches[1].values.tolist()
    unwatch = set(range(1, num_movies+1)) - set(watched)
    
    ns_list = random.sample(unwatch, 100)
    
    train_lst.append(watches.iloc[:-2])
    val_lst.append(watches.iloc[-2])
    test_lst.append(watches.iloc[-1])
    neg_lst.append(list(ns_list))


train = pd.concat(train_lst)
val   = pd.concat(val_lst, 1).T
test  = pd.concat(test_lst, 1).T

train.to_pickle('./data/ml/train_score.pkl')
val.to_pickle('./data/ml/val_score.pkl')
test.to_pickle('./data/ml/test_score.pkl')
np.save('./data/ml/neg_score.npy', neg_lst)