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
from utils.SaseFEdatasetFullclas import VideoSaseFEdatasetSingle
import os
from tqdm.auto import tqdm
import yaml
from utils.utils import *
from torch.utils.tensorboard import SummaryWriter
import wandb
import math
from utils.AverageMeter import AverageMeter
from utils.train_epochs import *
from utils.TripletLoss import *
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
            # model.load_state_dict(torch.load("checkpoints/trained_model.pth"))
            model.apply(init_weights)
            model = torch.jit.script(model).to(device)
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

            optimizer = init_optim(model, params["optim"], params["lr"], params["decay"])
            criterion = torch.jit.script(nn.TripletMarginLoss(margin=0.2))
             

            train_dataset, test_dataset = init_dataset(datamodule["dataset_dir"], datamodule["n_frames"], n_classes=num_classes, type="videos")
            int2emo = config["int2emo"]
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            # create path to save checkpoints

            if not os.path.exists("checkpoints/" + img_name):
                os.mkdir("checkpoints/" + img_name)
            #train the model
            model.train()
            for epoch in tqdm(range(params["max_epochs"]), desc="Epochs"):
                running_loss = []
                for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

                    anchor_img = anchor_img.to(device)
                    positive_img = positive_img.to(device)
                    negative_img = negative_img.to(device)
                    
                    optimizer.zero_grad()
                    anchor_out = model(anchor_img)
                    positive_out = model(positive_img)
                    negative_out = model(negative_img)
                    
                    loss = criterion(anchor_out, positive_out, negative_out)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss.append(loss.cpu().detach().numpy())
                print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epoch, np.mean(running_loss)))
                torch.save(model.state_dict(), "checkpoints/trained_model.pth")
                # if valid_loss< best_valid_loss:
                #     torch.save(model.state_dict(),'best_model.pt')
            
            train_results = []
            labels = []

            model.eval()
            with torch.no_grad():
                for img, _, _, label in tqdm(train_loader):
                    train_results.append(model(img.to(device)).cpu().numpy())
                    labels.append(label)
                    
            train_results = np.concatenate(train_results)
            labels = np.concatenate(labels)
            train_results.shape
            plt.figure(figsize=(15, 10), facecolor="azure")

            for label in np.unique(labels):
                tmp = train_results[labels==label]
                plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
            plt.legend()
            plt.savefig("results/train_results.png")