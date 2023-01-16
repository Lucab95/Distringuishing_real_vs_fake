import networkx as nx 
from skimage import io
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from utils.TripletDataset import TripletNetworkDataset
from utils.SASEFE_dataset import VideoSaseFEdatasetSingle
from networks.networks import *
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import time
import random
random.seed(42)

def plot_closest_imgs(anchor_images, anc_img_names, S_name, closest_idxs, distance, no_of_closest = 10):

    G=nx.Graph()

    # S_name = [img_path.split('/')[-1]]

    for s in range(no_of_closest):
        S_name.append(anc_img_names.iloc[closest_idxs[s]])

    for i in range(len(S_name)):
        image =anchor_images[i]
        G.add_node(i,image = image)
        
    for j in range(1,no_of_closest + 1):
        G.add_edge(0,j,weight=distance[closest_idxs[j-1]])
        

    pos=nx.kamada_kawai_layout(G)

    fig=plt.figure(figsize=(20,20))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    ax.set_title(S_name[n][0:4])
    ax.axis('off')
    ax.axis('off')
    plt.show()

def init_dataset(DATA_FOLDER,n_frames,n_classes=12,test_size = 10,type= "triplet"):
    # add_transforms = [transforms.Resize(224), transforms.Normalize([0.454, 0.390, 0.331], [0.164, 0.187, 0.152])]
    start = time.time()
    add_transforms = [transforms.Resize(224)]# , transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    x= np.arange(0, 50, 1, dtype=int)
    np.random.shuffle(x)
    
    if type== "triplet":
        print("Loading dataset...")
        test_subj = x[:40]
        train_dataset= TripletNetworkDataset(DATA_FOLDER, n_emt=n_classes, required_frames=n_frames,transforms=add_transforms, test_subj=test_subj,train=True)
        test_dataset = TripletNetworkDataset(DATA_FOLDER, n_emt=n_classes, required_frames=n_frames,transforms=add_transforms, test_subj=test_subj,train=False)
    else:
        test_val = [1, 15, 16, 41, 44, 17, 21, 26, 30, 37]
        train_dataset = VideoSaseFEdatasetSingle(DATA_FOLDER, n_emt=n_classes, required_frames=n_frames,transforms=add_transforms, test_subj=test_val,train_test=0, augment=False)
        val_dataset = VideoSaseFEdatasetSingle(DATA_FOLDER, n_emt=n_classes, required_frames=n_frames,transforms=add_transforms, test_subj=test_val,train_test=1)
        test_dataset = VideoSaseFEdatasetSingle(DATA_FOLDER, n_emt=n_classes, required_frames=n_frames,transforms=add_transforms, test_subj=test_val,train_test=2)
        return train_dataset,val_dataset,test_dataset

    print("Dataset loaded", time.time()-start)
    return train_dataset, test_dataset
    
def init_model(model_name: str, num_classes: int, device: str, embeddings_size: int = 512):
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
    if model_name == "resnet18":
        # model = MyResNet()
        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(512, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "cnn": 
        model = CNN_Model(emb_size=embeddings_size)
    elif model_name == "ResNet50":
        from networks.resnet import resnet50
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False,)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
)
        # model= resnet50(pretrained=True, num_classes=num_classes)
    elif model_name == "vgg-lstm":
        from networks.vgg_lstm import VGGLSTM
        model = VGGLSTM(num_classes=num_classes)
    elif model_name == "cnn-transformer":
        from networks.cnn_transformer import CnnTransformer
        cnn_ckpt="checkpoints\\cnn_transformer\\resnet50_ft_weight.pkl"
        model = CnnTransformer( num_patch = 9, embed_dim= 768, output_dim= num_classes, num_heads=4, dropout=0.5, cnn_ckpt=cnn_ckpt)

    else:
        model = Network(emb_dim=embeddings_size)

        # model = models.resnet152(pretrained=False)
        # model.conv1 = torch.nn.Conv2d(
        #     3, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # model.fc = nn.Linear(2048, num_classes)

    model.to(device)
    return model

def get_embeddings(model,dataset, device):
  model.eval()
  embeddings = []
  # targets = []
  with torch.no_grad():
    for video in dataset:
      images,anch_lab, neg_lab= video
      # print(np.array(images).shape)
      images = images.to(device)
      # embeddings.append([model(images),anch_lab])

      # break
      # for frame in images:
      #   targets.append(anch_lab)
      #   print(frame.shape)
      #   frame = frame.to(device)
        
      #   emb = model(frame.unsqueeze(0))
      #   print(np.array((frame)).shape)
      #   break
      #   embeddings.append((emb.cpu().numpy(),anch_lab))
      # targets = np.full((len(images)),anch_lab)
      embeddings.append([model(images),anch_lab])
  return embeddings

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

