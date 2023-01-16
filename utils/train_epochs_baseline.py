import csv
# from AverageMeter import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.utils import *
import torch
from tqdm import tqdm
import math

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
    best_train_loss = [999, -1]
    best_val_loss = [999, -1]
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    n_steps_per_epoch = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    example_ct = 0
    print("Start epoch #%d" % (epoch))
    for batch_idx, inputs in tqdm(enumerate(data_loader)):
        running_loss = 0.0
        correct = 0
        (data, targets) = inputs
        targets = targets.type(torch.LongTensor)
        data = data.squeeze(0)
        targets=targets.repeat(data.shape[0])
        
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                "Progress: {:d}/{:d}. Loss:{:.3f}".format(
                    batch_idx, len(data_loader), running_loss / (batch_idx + 1)
                )
            )

        example_ct += len(inputs)
        metrics = {
            "train/train_loss": loss,
            "train/epoch": (batch_idx + 1 + (n_steps_per_epoch * epoch))
                / n_steps_per_epoch,
            "train/example_ct": example_ct,
        }





    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg 

def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            targets = targets.type(torch.LongTensor)
            data = data.squeeze(0)
            targets=targets.repeat(data.shape[0])
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)  

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg


def train_fn(model,dataloader, optimizer, criterion, device):
  model.train()
  total_loss = 0.0

  for A,P,N in tqdm(dataloader):
    A,P,N = A.to(device), P.to(device), N.to(device)
    A_embs = model(A)
    P_embs = model(P)
    N_embs = model(N)

    loss = criterion(A_embs,P_embs,N_embs)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss / len(dataloader)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)