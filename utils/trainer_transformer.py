import torch
from torch.utils.data import Subset, DataLoader, random_split
import math
import torchvision.transforms as transforms
from tqdm.auto import tqdm
transform = transforms.ToPILImage()
import wandb
from utils.UnNormalize import UnNormalize
from utils.train_epochs_transformer import train_epoch, val_epoch, validate, train
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn.functional as Func
import numpy as np


def train_tf(
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
        correct = 0

        print("\n\n -- Training --")
        model.train()
        correct_predictions = 0

        train_loss, acc = train(model, train_loader, optimizer, criterion, device)
        # print("Train loss: ", train_loss)
        # print("Train pred: ", train_prediction)
        # print("Train labels: ", labels)
        # wandb.log({"train_loss": train_loss})
        print("Train loss: ", train_loss)
        print("Train accuracy: ", acc)
        # wandb.log({"train_loss": train_loss})
            



            # if epoch == 1 and i == log_video:
            #     unnorm = UnNormalize(mean=[131.0912, 103.8827, 91.4953], std=[1, 1, 1])
            #     for one_frame in inputs:
            #         log_input.append(one_frame)
            #     wandb.log(
            #         {"input_example": [wandb.Image(image) for image in log_input]}
            #     )
            #     wandb.log({"input_size": inputs.size()})
            #     # print(inputs.size())
            #     print("logged_input")
                # model.conv1 = nn.Conv2d(inputs.size(dim=1), 64, kernel_size=7, stride=2, padding=3, bias=False)
        # if i % 100 == 0:
        #     print(
        #         "Progress: {:d}/{:d}. Loss:{:.3f}".format(
        #             i, len(train_loader), running_loss / (i + 1)
        #         )
        #     )
        # example_ct += len(inputs)
        metrics = {
            "train/train_loss": train_loss,
            "train/train_accuracy": acc,
            "train/epoch": epoch,
        }
        #end train part

        if train_loss < best_train_loss[0]:
            # keep track of best train loss
            best_train_loss[0] = train_loss
            best_train_loss[1] = epoch

        if abs(train_loss - prev_epoch_loss) < 0.001:
            print(
                "Early stopping! Epoch: {:d}, Loss: {:.3f}".format(epoch, train_loss)
            )
            stop_count += 1
            if stop_count == 3:
                last_epoch = True
            prev_epoch_loss = train_loss
        else:
            prev_epoch_loss = train_loss
            stop_count = 0

        val_loss, val_prediction, labels = validate(
            model, val_loader, device, criterion
        )
        # print("Val loss: ", val_loss)
        # print("Val pred: ", val_prediction)
        # print("Val labels: ", labels)
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
        accuracy = correct_predictions / len(train_loader.dataset)
        print("Accuracy: {:.3f}".format(accuracy))

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
