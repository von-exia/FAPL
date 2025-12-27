from sklearn import metrics
import torch

def compute_AP(y, pred, n_class=1):
    ## compute one score
    if n_class == 1:
        auc = metrics.average_precision_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc

def compute_AUC(y, pred, n_class=1):
    ## compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc

def compute_F1(y, pred, n_class=1, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        f1 = metrics.f1_score(y, pred)
        return f1
    else:
        ## compute two-class
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        f1 = metrics.f1_score(y, index)
    
    return f1

def compute_recall(y, pred, n_class=1, t=0.5):
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        recall = metrics.recall_score(y, pred)
        return recall
    else:
        index = torch.argmax(pred, dim=1)
        recall = metrics.recall_score(y_true=y, y_pred=index)
        return recall

def compute_precision(y, pred, n_class=1, t=0.5):
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        pre = metrics.precision_score(y, pred)
        return pre
    else:
        index = torch.argmax(pred, dim=1)
        pre = metrics.precision_score(y_true=y, y_pred=index)
    return pre

def compute_ACC(y, pred, n_class=2, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        acc = metrics.accuracy_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        acc = metrics.accuracy_score(y, index)
        # acc = metrics.f1_score(y, index)
    
    return acc