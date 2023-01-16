import torch
from torch.utils.data import Subset, DataLoader, random_split
import math
import torchvision.transforms as transforms
from tqdm.auto import tqdm
transform = transforms.ToPILImage()
import wandb
from utils.UnNormalize import UnNormalize
from utils.train_epochs import train_epoch, val_epoch, validate
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn.functional as Func
import numpy as np
def train(
    model,
    criterion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim,
    step: int,
    max_epochs: int,
    model_name: str,
    classes: list,
    one_hot= False
):

    prev_epoch_loss = 0.0

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step, gamma=0.1, verbose=True
    )

    last_epoch = False
    stop_count = 0

    best_train_loss = [999, -1]
    best_val_loss = [999, -1]
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
    example_ct = 0
    conf_labels = []
    conf_pred = []
    log_input = []
    log_video = random.randint(0, len(train_loader.dataset)-1)
    for epoch in range(1, max_epochs + 1):
        print("Start epoch #%d" % (epoch))
        running_loss = 0.0
        correct = 0

        print("\n\n -- Training --")
        model.train()
        for i, batch in enumerate(tqdm(train_loader, 0)):

            inputs,labels = batch
            labels = labels.type(torch.LongTensor)
            if one_hot:
                labels = Func.one_hot(labels, num_classes = len(classes))
            inputs,labels = inputs.to(device), labels.to(device)
            


            if epoch == 1 and i == log_video:
                unnorm = UnNormalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])
                for one_frame in batch[0][0]:
                    log_input.append(one_frame)
                wandb.log(
                    {"input_example": [wandb.Image(image) for image in log_input]}
                )
                wandb.log({"input_size": inputs.size()})
                # print(inputs.size())
                print("logged_input")
                # model.conv1 = nn.Conv2d(inputs.size(dim=1), 64, kernel_size=7, stride=2, padding=3, bias=False)

            optimizer.zero_grad()

            outputs = model(inputs)
            print(outputs,labels)
            if outputs.shape[-1] == 1:
                labels = labels.unsqueeze(1)
            print(outputs,labels)
            topk=(1,)
            maxk = max(topk)
            batch_size = labels.size(0)
            
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            print(correct)
            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res









            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(
                    "Progress: {:d}/{:d}. Loss:{:.3f}".format(
                        i, len(train_loader), running_loss / (i + 1)
                    )
                )
            example_ct += len(inputs)
            metrics = {
                "train/train_loss": loss,
                "train/epoch": (i + 1 + (n_steps_per_epoch * epoch))
                / n_steps_per_epoch,
                "train/example_ct": example_ct,
            }
        running_loss = running_loss / len(train_loader)

        if running_loss < best_train_loss[0]:
            # keep track of best train loss
            best_train_loss[0] = running_loss
            best_train_loss[1] = epoch

        if abs(running_loss - prev_epoch_loss) < 0.001:
            print(
                "Early stopping! Epoch: {:d}, Loss: {:.3f}".format(epoch, running_loss)
            )
            stop_count += 1
            if stop_count == 3:
                last_epoch = True
            prev_epoch_loss = running_loss
        else:
            prev_epoch_loss = running_loss
            stop_count = 0

        val_loss, val_prediction, labels = validate(
            model, val_loader, device, criterion
        )
        if one_hot:
            labels= np.argmax(labels)
        correct += (val_prediction == labels).sum().item()

        val_metrics = {
            "val/val_loss": val_loss,
            "val/val_accuracy": correct / len(val_loader.dataset),
        }
        print(val_metrics)

        wandb.log({**metrics, **val_metrics})

        # check the loss to save best model and values for conf_matrix
        if val_loss < best_val_loss[0]:
            best_val_loss[0] = val_loss
            best_val_loss[1] = epoch
            conf_labels = labels
            conf_pred = val_prediction
            torch.save(
                model.state_dict(), "checkpoints/" + model_name + "_best.pt"
            )

        # save every 3epochs
        if epoch % 3 == 0:
            torch.save(
                model.state_dict(),
                "checkpoints/" + model_name + "_epoch" + str(epoch) + ".pt",
            )

        scheduler.step()

        if last_epoch:
            break

    # create confusion matrix on Wandb
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None, y_true=conf_labels, preds=conf_pred, class_names=classes
            )
        }
    )
    wandb.finish()
    return best_train_loss, best_val_loss
