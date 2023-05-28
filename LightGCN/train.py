import torch
import numpy as np
from utils.metrics import get_test, AverageMeter
import dgl

def train(train_loader, model, optimizer, args, mask=None):
    model.train()

    avg_loss = AverageMeter()
    avg_re_loss = AverageMeter()
    avg_reg_loss = AverageMeter()

    for i, batch in enumerate(train_loader):
        user, item, item_j = batch
        bsz = len(user)
        loss, reconstruct_loss, reg_loss =model(user.to(torch.device(args.gpu)),
                                                item.to(torch.device(args.gpu)),
                                                item_j.to(torch.device(args.gpu)))
        
        optimizer.zero_grad()
        loss.backward()

        if mask is not None: mask.step()        
        else: optimizer.step()                  


        avg_loss.update(loss.item(), bsz)
        avg_re_loss.update(reconstruct_loss.item(), bsz)
        avg_reg_loss.update(reg_loss.item(), bsz)
    return avg_loss.avg, avg_re_loss.avg, avg_reg_loss.avg

def test(test_loader, model, item_min, user_seq, args):
    model.eval()
    z = model.get_emb()
    item_emb = z[item_min:]

    rating_list = []
    groundTrue_list = []
    max_K = max(args.topks)

    for batch in test_loader:
        user_id, groundTrue = batch
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  
        rec = rec.detach().cpu().numpy()

        rating_list.append(rec)
        groundTrue_list.append(groundTrue)

    result = get_test(rating_list, groundTrue_list, args)
    for k, v in result.items():
        result[k] = v[0]
    return result




# according user
def group_test_user(test_loader, model, item_min, user_seq, args, group_u_num, group_i_num, user_group, item_group):
    model.eval()
    z = model.get_emb()
    item_emb = z[item_min:]

    result = []
    userid = torch.arange(0, item_min, 1)
    max_K = max(args.topks)
    for i in range(len(group_u_num)):
        user_id = userid[user_group == i]
        groundTrue = [user_seq[i] for i in user_id.tolist()]
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user.item()]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  
        rec = rec.detach().cpu().numpy()

        res = []
        for k, v in get_test([rec], [groundTrue], args).items():
            res.append(v[0])

        result.append(res)  

    print("user_group_precision:", end=" ")
    print([result[i][0] for i in range(len(group_u_num))])

    print("user_group_recall:", end=" ")
    print([result[i][1] for i in range(len(group_u_num))])
    return result


# according item
def group_test_item(test_loader, model, item_min, user_seq, args, group_u_num, group_i_num, user_group,
                    item_group):  
    model.eval()
    z = model.get_emb()
    item_emb = z[item_min:]

    rating_list = [[] for _ in range(len(group_i_num))]
    groundTrue_list = []

    max_K = max(args.topks)

    for batch in test_loader:
        user_id, groundTrue = batch
        user_emb = z[user_id]
        ranking_score = user_emb.matmul(item_emb.t())

        for idx, user in enumerate(user_id):
            ranking_score[idx][np.array(user_seq[user]) - item_min] = -np.inf

        _, rec = torch.topk(ranking_score, max_K)  # return the idx
        rec = rec.detach().cpu().numpy()
        rating = [[[] for _ in range(len(user_id))] for _ in range(len(group_i_num))]
        for i in range(len(user_id)):
            for r in rec[i]:
                # item_group[i - item_min]  belongs to ~ group
                rating[item_group[r]][i].append(r)

        for i in range(len(rating)):
            for j in range(len(rating[i])):
                rating[i][j].extend([[-1]] * (20 - len(rating[i][j])))
            rating_list[i].append(np.array(rating[i]))

        groundTrue_list.append(groundTrue)

    res = []
    for i in range(len(group_i_num)):
        result = []
        r = get_test(rating_list[i], groundTrue_list, args)
        for k, v in r.items():
            result.append(round(v[0], 6))
        res.append(result)

    print("item_group_precision:", end=" ")
    print([res[i][0] for i in range(len(group_i_num))])

    print("item_group_recall:", end=" ")
    print([res[i][1] for i in range(len(group_i_num))])

    # recall_whole = sum([res[i][1] for i in range(10)])
    # precision_whole = sum([res[i][0] for i in range(10)])
    return res