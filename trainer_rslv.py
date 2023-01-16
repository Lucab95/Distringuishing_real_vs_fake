import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import wandb
wandb.login()
# f7d09dd4e236f76cbd35c414e968ae2c33de7074

random.seed(42)
transform = T.ToPILImage()

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 0
classes = []
le = preprocessing.LabelEncoder()


def init_model(model_name: str, img_src: list):
    """Create the model and change the first layer and final layer

    Args:
        model_name (str): type of model to initialize
        img_src (list): list of image sources for the model,
                        used to change the input channels of the model

    Returns:
        model (torch.nn): return the modified model
    """

    # try resnet 18
    print("model used:", model_name)
    input_chann = 0
    for idx, src in enumerate(img_src):
        if img_src[idx] == "graphonomy":
            input_chann += 1
        else:
            input_chann += 3
    if model_name == "resnet18":
        # model = MyResNet()
        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_chann, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(512, num_classes)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            input_chann, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(2048, num_classes)

    model.to(device)
    return model


def init_dataset(dm_config: dict, size: int = 0):
    """Initalize the dataset and sets number of classes and classes name, also prepare the encoder.

    Args:
        dm_config (dict): dict to inizialize the datamodule
        size (int, optional): take subsample of the dataframe. Defaults to 0.

    Returns:
        train_loader (DataLoader): train dataloader
        val_loader (DataLoader): validation dataloader
    """

    datamodule = VtonDataModule(**dm_config)
    dset = datamodule.predict_dataset

    global classes
    classes = dset.df["manual_label"].unique()
    le.fit(classes)

    global num_classes
    num_classes = len(dset.df["manual_label"].unique())
    # subsample

    if size != 0:
        # shuffle
        dset.df = dset.df.sample(frac=1).reset_index(drop=True)
        dset.df.head(size)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    return train_loader, val_loader  # train_dataset, val_dataset


def init_optim(model: torch.nn, optim: torch.optim, lr: float, decay: float):
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise TypeError("Optim %s is not allowed." % optim)


def process_tensor_src(batch: int, img_src: list):
    """
    concatenate tensors if the image srcs are more than 1,
    channel-wise concatenation

    Args:
        batch (int): batch size
        img_src (list): list of images name to concatenate

    Returns:
        inputs (torch.Tensor): tensor with channel stacked images

    """
    inputs = batch[img_src[0]]
    for idx in range(1, len(img_src)):
        # print(inputs.size(), batch[img_src[idx]].size())
        inputs = torch.cat((inputs, batch[img_src[idx]]), 1)
    return inputs


def create_name(img_src: list):
    """This function creates a unique name for saving
    and keep track fo the experiments, simply join the different
    img sources in one string

    Args:
        img_src (list): list of images used for the training

    Returns:
        img_name (str): a concatenated string
    """
    img_name = "_".join(img_src)
    if len(img_src) > 1:
        img_name = ""
        for method in img_src:
            img_name += method[:4]
    return img_name


def validate(model, val_loader: DataLoader, image_src=str, criterion=torch.nn):
    print("Validation")
    model.eval()

    with torch.no_grad():
        test_loss = 0.0
        pred = []
        labels = []
        for i, batch in enumerate(val_loader, 0):
            # check if needs to concatenate image or not
            inputs = process_tensor_src(batch, img_src)
            label = batch["label"]

            label = le.transform(label)
            label = torch.tensor(label)

            if torch.cuda.is_available():
                inputs = inputs.to(device)
                label = label.to(device)

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


def train(
    model,
    criterion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    image_src: np.array,
    optimizer: torch.optim,
    step: int,
    max_epochs: int,
    model_name: str,
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
    for epoch in range(1, max_epochs + 1):
        print("Start epoch #%d" % (epoch))
        running_loss = 0.0
        correct = 0

        print("\n\n -- Training --")
        model.train()
        for i, batch in enumerate(tqdm(train_loader, 0)):

            # log input
            # check if needs to concatenate image or not
            inputs = process_tensor_src(batch, img_src)
            labels = batch["label"]

            if epoch == 1 and i == 0:
                for src in img_src:
                    log_input.append(transform(batch[src][0]))
                wandb.log(
                    {"input_example": [wandb.Image(image) for image in log_input]}
                )
                wandb.log({"input_size": inputs.size()})
                # print(inputs.size())
                print("logged_input")
                # model.conv1 = nn.Conv2d(inputs.size(dim=1), 64, kernel_size=7, stride=2, padding=3, bias=False)

            # get encoding of labels
            labels = le.transform(labels)
            labels = torch.from_numpy(labels)
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            if outputs.shape[-1] == 1:
                labels = labels.unsqueeze(1)

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
            model, val_loader, image_src, criterion
        )
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
                model.state_dict(), "/app/checkpoints/" + model_name + "_best.pt"
            )

        # save every 3epochs
        if epoch % 3 == 0:
            torch.save(
                model.state_dict(),
                "/app/checkpoints/" + model_name + "_epoch" + str(epoch) + ".pt",
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


if __name__ == "__main__":

    config_path = "/app/configs/template_train.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # make params accessible by dot notation
    params = config["training_params"]

    train_loader, val_loader = init_dataset(config["datamodule"], params.size)

    for net in config["models"]:
        for img_src in config["img_srcs"]:

            # initialize the model
            model = init_model(net, img_src)
            img_name = create_name(img_src)
            # Folder to save
            logdir = os.path.join(img_name, net)
            print("\n Training arguments:\n", params, net, img_name)
            print("\nModel log dir:", logdir, "\n")

            optim = init_optim(model, params.optim, params.lr, params.decay)
            criterion = torch.nn.CrossEntropyLoss()

            # We define a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)

            wandb_name = (net + "_" + img_name + "_ep_" + str(params.max_epochs),)
            wandb.init(
                # Set the project where this run will be logged
                project="autolabeller",
                name=f"{wandb_name}",
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": params.lr,
                    "model": net,
                    "dataset_size": params.size,
                    "image_src": img_name,
                    "epochs": params.max_epochs,
                },
            )

            # create path to save checkpoints
            if not os.path.exists("/app/checkpoints/" + img_name):
                os.mkdir("/app/checkpoints/" + img_name)

            train_loss, val_loss = train(
                model,
                criterion,
                train_loader,
                val_loader,
                img_src,
                optim,
                params.step,
                params.max_epochs,
                logdir,
            )

            print("Best train loss:", train_loss)
            print("Best val loss:", val_loss)