from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np
import sys
import sys
sys.path.append('..')
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from utils.TripletDataset import TripletNetworkDataset
import os
from tqdm.auto import tqdm
import yaml
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
import wandb
import math
from utils.AverageMeter import AverageMeter
from utils.train_epochs_baseline import train_epoch, val_epoch
transform = transforms.ToPILImage()
input_type= "video"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# wandb.login()
# random.seed(42)
# wandb.tensorboard.patch(save=False, tensorboardX=True)


    # create confusion matrix on Wandb
    # wandb.log(
    #     {
    #         "conf_mat": wandb.plot.confusion_matrix(
    #             probs=None, y_true=conf_labels, preds=conf_pred, class_names=classes
    #         )
    #     }
    # )
    # wandb.finish()
    # return best_train_loss, best_val_loss

def init_optim(model: torch.nn, optim: torch.optim, lr: float, decay: float):
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise TypeError("Optim %s is not allowed." % optim)

if __name__ == "__main__":

    config_path = "configs/template_train_baseline.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # make params accessible by dot notation
    params = config["training_params"]
    datamodule = config["datamodule"]
    num_classes = params["num_classes"]
    train_loader, val_loader = init_dataset(datamodule["dataset_dir"],datamodule["n_frames"], num_classes, test_size=10)

    for net in config["models"]:
            # initialize the modelsource            
            model = init_model(net, num_classes, device, embeddings_size=512)
            
            # model.load_state_dict(torch.load("models/best_model.pt"))
            img_name = net+"_"+str(num_classes)

            # We define a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # wandb_name = (img_name + "_ep_" + str(params["max_epochs"]))
            # wandb.init(
            #     # Set the project where this run will be logged
            #     project="Distinguish",
            #     name=f"{wandb_name}",
            #     # Track hyperparameters and run metadata
            #     config={
            #         "learning_rate": params["lr"],
            #         "model": net,
            #         "dataset_size": params["size"],
            #         "image_src": img_name,
            #         "epochs": params["max_epochs"],
            #     },
            # )
            # Folder to save
            logdir = os.path.join("models", img_name, net)
            print("\n Training arguments:\n", params, net, img_name)
            print("\nModel log dir:", logdir, "\n")

            optim = init_optim(model, params["optim"], params["lr"], params["decay"])
            criterion = torch.nn.CrossEntropyLoss()
             

            train_dataset, val_dataset, test_dataset = init_dataset(datamodule["dataset_dir"], datamodule["n_frames"], n_classes=num_classes,type= "baseline")
            int2emo = config["int2emo"]
            print("get embeddings")
            # embeddings = get_embeddings(model,train_dataset, device)
            # val_embeddings = get_embeddings(model,test_dataset, device)
            

            print("get embeddings done")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            # create path to save checkpoints
            if not os.path.exists("checkpoints/" + img_name):
                os.mkdir("checkpoints/" + img_name)
            #classifyer part
            for epoch in range(0, params["max_epochs"]):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optim, epoch, log_interval=10, device=device)
                val_loss, val_acc = val_epoch(model, test_loader, criterion, device)
                # scheduler.step(val_loss)
                # write summary
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
                # torch.save(state, os.path.join('snapshots', f'{"model-lst"}-Epoch-{epoch}-Loss-{val_loss}.pth'))
                if (epoch) % 5 == 0:
                    print("Epoch {} model saved!\n".format(epoch))

            print("Best train loss:", train_loss)
            print("Best val loss:", train_acc)