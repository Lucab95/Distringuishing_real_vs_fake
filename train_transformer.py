import networkx as nx 
from skimage import io
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from networks.networks import *
import torch
import torch.nn as nn
import numpy as np

import yaml
from utils.utils import *
import random
from PIL import Image
import numpy as np
import sys
sys.path.append('..')
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import yaml
from utils.utils import *
import wandb
from utils.AverageMeter import AverageMeter
from dict_emotions import emo_fake_true_12,emofake_true,emo_basic_6
input_type= "video"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)
transform = transforms.ToPILImage()
from utils.trainer_lstm import *

os.system("wandb login f7d09dd4e236f76cbd35c414e968ae2c33de7074")

def init_optim(model: torch.nn, optim: torch.optim, lr: float, decay: float):
    if optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise TypeError("Optim %s is not allowed." % optim)



if __name__ == "__main__":

    config_path = "configs/template_train_cnn_transformer.yaml"
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # make params accessible by dot notation
    params = config["training_params"]
    datamodule = config["datamodule"]
    num_classes = params["num_classes"]

    train_dataset, val_dataset, test_dataset= init_dataset(datamodule["dataset_dir"],datamodule["n_frames"], num_classes, test_size=5, type=datamodule["type"])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    classes = []
    if num_classes==12:
        int2emo = emo_fake_true_12
    elif num_classes==2:
        int2emo = emofake_true
    elif num_classes==6:
        int2emo = emo_basic_6
    for key in int2emo:
        classes.append(int2emo[key])

for net in config["models"]:
        # initialize the modelsource            
        model = init_model(net, num_classes, device, embeddings_size=512)
        model.to(device)
        # model.load_state_dict(torch.load("models/best_model.pt"))
        img_name = net+"_"+str(num_classes)

        # We define a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        wandb_name = (img_name + "_ep_" + str(params["max_epochs"]))
        wandb.init(
            # Set the project where this run will be logged
            project="Distinguish",
            name=f"{wandb_name}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": params["lr"],
                "model": net,
                "dataset_size": params["size"],
                "image_src": img_name,
                "epochs": params["max_epochs"],
            },
            mode="disabled",
        )
        # Folder to save
        logdir = img_name+"/"+net
        print("\n Training arguments:\n", params, net, img_name)
        print("\nModel log dir:", logdir, "\n")

        optim = init_optim(model, params["optim"], params["lr"], params["decay"])
        criterion = torch.nn.CrossEntropyLoss()
            

        train_dataset, val_dataset, test_dataset,  = init_dataset(datamodule["dataset_dir"], datamodule["n_frames"], n_classes=num_classes, test_size=10, type=datamodule["type"])
        int2emo = config["int2emo"]
        
        # if datamodule["type"] == "frames":
        #     train_frames=[]
        #     val_frames=[]
        #     for video, label in train_dataset:
        #         print(label)
        #         for vid in video:
        #             train_frames.append([vid,label])
        #     for video, label in val_dataset:
        #         for vid in video:
        #             val_frames.append([vid,label])
        #     train_dataset = train_frames
        #     val_dataset = val_frames

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
        # create path to save checkpoints
        if not os.path.exists("checkpoints/" + img_name):
            os.mkdir("checkpoints/" + img_name)
        #classifyer part
        print(classes)
        train_loss, val_loss = train(
                        model,
                        criterion,
                        train_loader,
                        val_loader,
                        optim,
                        params["step"],
                        params["max_epochs"],
                        logdir,
                        classes,
                        one_hot=True
                    )

        print("Best train loss:", train_loss)
        print("Best val loss:", val_loss)

    
