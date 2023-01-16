import csv
import numpy as np
from sklearn.metrics import f1_score
import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def EXPR_metric(x, y):
    # x: predict; y: target
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return f1, acc, 0.67*f1 + 0.33*acc

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True)
    # _, pred = output.topk(maxk, 1, True, True)
    # pred = torch.squeeze(pred,0)
    pred = pred.t()
    print(pred)
    # x = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(target.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size
    # correct = pred.eq(target)
    # return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

    # batch_size = targets.size(0)

    # _, pred = outputs.topk(1, 1, True)
    # pred = pred.t()
    # correct = pred.eq(targets.view(1, -1))
    # n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size