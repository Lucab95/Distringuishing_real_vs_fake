import csv
# from AverageMeter import *
from utils.AverageMeter import AverageMeter
from utils.Logger import Logger
from utils.utils import *
import torch
from tqdm import tqdm

def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
 
    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader)):
        
        targets = targets.type(torch.LongTensor)
        data, targets = data.to(device), targets.to(device)
        
        outputs = model(data)
        
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))


    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg  