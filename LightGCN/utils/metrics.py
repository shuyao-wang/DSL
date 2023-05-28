import numpy as np
from sklearn.metrics import roc_auc_score
import multiprocessing
from functools import partial


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """

    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


# ====================end Metrics=============================
# =========================================================


def test_one_batch(X, topk):
    sorted_items = X[0]
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topk:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def get_test(rating_list, groundTrue_list, args):
    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}
    pool = multiprocessing.Pool(1)

    non_empty_user_num = 0
    non_empty_groundTrue = []
    non_empty_rating = []
    for batch_id in range(len(groundTrue_list)):
        non_empty_groundTrue.append([])
        non_empty_rating.append([])
        for ins_id in range(len(groundTrue_list[batch_id])):
            if len(groundTrue_list[batch_id][ins_id]) > 0:
                non_empty_groundTrue[batch_id].append(groundTrue_list[batch_id][ins_id])
                non_empty_rating[batch_id].append(rating_list[batch_id][ins_id])
                non_empty_user_num += 1

    non_empty_groundTrue = list(filter(None, non_empty_groundTrue))
    non_empty_rating = list(filter(None, non_empty_rating))

    X = zip(non_empty_rating, non_empty_groundTrue)
    par = partial(test_one_batch, topk=args.topks)  # PROCESS MANY param
    pre_results = pool.map(par, X)

    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']

    results['recall'] /= float(non_empty_user_num)
    results['precision'] /= float(non_empty_user_num)
    results['ndcg'] /= float(non_empty_user_num)

    pool.close()
    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
