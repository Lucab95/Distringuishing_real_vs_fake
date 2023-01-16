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
    for batch_idx, (data, targets, targets_neg) in tqdm(enumerate(data_loader)):
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

def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            targets = targets.type(torch.LongTensor)
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg


def validate(model, val_loader,device, criterion):
    print("Validation")
    model.eval()

    with torch.no_grad():
        test_loss = 0.0
        pred = []
        labels = []
        for i, batch in enumerate(val_loader, 0):
            # check if needs to concatenate image or not
            inputs,label = batch
            label = label.type(torch.LongTensor)
            inputs,label = inputs.to(device), label.to(device)


            outputs = model(inputs)
            if outputs.shape[-1] > 1:
                o_labels = torch.argmax(outputs, dim=1)
            else:
                label = label.unsqueeze(1)
                o_labels = torch.round(outputs)
            pred.append(o_labels.cpu())

            labels.append(label.cpu())

            loss = criterion(outputs, o_labels)

            test_loss += loss.item()
            if i % 100 == 0:
                print("Progress: {:d}/{:d}".format(i, len(val_loader)))

        pred = torch.cat(pred).numpy()
        labels = torch.cat(labels).numpy()
        val_loss = test_loss / len(val_loader)

        # print("Validation loss: {:f}".format(val_loss))
        # print("Confusion matrix")
        # print(confusion_matrix(labels, pred))
        # print(classification_report(labels, pred, labels =np.arange(num_classes), target_names=classes))

        return val_loss, pred, labels

def train(model, train_loader,optimizer, criterion, device):
    optimizer.zero_grad()
    train_loss = 0.0
    pred = []
    labels = []
    correct_predictions = 0
    for i, batch in tqdm(enumerate(train_loader, 0)):

        inputs,label = batch
        label = label.type(torch.LongTensor)
        inputs,label = inputs.to(device), label.to(device)

        outputs = model(inputs)
        probs = torch.exp(outputs)
        print("outp",outputs)
        print(probs, )
        _, preds = probs.max(1)
        print("preds",preds)
        if outputs.shape[-1] > 1:
            o_labels = torch.argmax(outputs, dim=1)
        else:
            label = label.unsqueeze(1)
            o_labels = torch.round(outputs)
        pred.append(o_labels.cpu())

        # labels.append(label.cpu())
        correct_predictions += (preds == labels).sum().item()
        loss = criterion(outputs, o_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    pred = torch.cat(pred).numpy()
    labels = torch.cat(labels).numpy()
    train_loss_final = train_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader)
    return train_loss_final, accuracy
