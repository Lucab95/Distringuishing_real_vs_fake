from logging import raiseExceptions
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import os
from pyrsistent import v
import torch
import torchvision
from torch.utils.data import Dataset, Sampler
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, RandomRotation, ColorJitter
import cv2
import re
import Video2frame
from dict_emotions import emo_fake_true_12,emofake_true,emo_basic_6
# from dataset import SaseFEdataset
PROJ_PATH = os.path.abspath(os.getcwd())
db_path = PROJ_PATH +"\Output\\"
import torch
'cuda' if torch.cuda.is_available() else 'cpu'

try:
  import google.colab
  colab_idx = 1
except:
  colab_idx = 3

class VideoSaseFEdatasetSingle(Dataset):
    def __init__(self,video_folder, classtype="faketrue12emo", tensor = True, required_frames = 1,transforms=None):
        self.video_folder = video_folder
        self.classtype = classtype
        self.files_path = []
        self.labels = []
        self.subject = []
        self.tensor = tensor
        default_transforms = [ToTensor()]
        if transforms is not None:
            default_transforms += transforms
        self.transforms = Compose(default_transforms)
        self.K = 50
        self.get_file_paths()
        self.files_path= sorted(self.files_path, key=lambda i: int((re.split('([0-9]+)',i)[colab_idx])))
        self.get_file_labels()
        self.indexes = np.arange(0,self.K,self.K /required_frames).astype(int)

    def get_file_paths(self): 
        # split to train / val
        format = "k"+ str(self.K)
        for root, d, files in os.walk(self.video_folder):
            for f in files:
                if f.endswith(format):
                    fpath = os.path.join(root, f)
                    self.files_path.append(fpath)
                
        

    def get_file_labels(self):
        self.labels = [Path(f).parts[-2] for f in self.files_path]
        self.subject =  [int(re.split('([0-9]+)',f)[colab_idx]) for f in self.files_path]
        return self.labels

    def get_shape(self,idx):
        print("tensor shape:", self[idx][0].detach().cpu().numpy().shape, "\nlabel shape:", self[idx][1].detach().cpu().numpy().shape)


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (v_frames, label) where label is the emotion of the video
        """


        with open(self.files_path[idx], "rb") as f:
            v_frames = np.load(f)  # load all the frames having this shape: (T, H, W, C) T=frame number

        # get the necessary indexes of frames for the video
        v_frames= v_frames[self.indexes].astype(np.uint8)
        # .astype(np.float32)
        

        #apply the custom transformations
        if self.transforms:
            res_vframes = []
            for i in range(len(v_frames)):
                res_vframes.append(self.transforms((v_frames[i])))
            res_vframes = torch.stack(res_vframes, 0)
        

        label = self.__get_categorical_label(idx)
        subject = re.split('([0-9]+)',self.files_path[idx])[colab_idx]

        
        if self.tensor:
            v_frames = res_vframes
            label = torch.tensor(label)
        else:
            v_frames = np.stack(v_frames, 0)
        
        return v_frames, label, int(subject)
        # res_vframes = res_vframes.permute(1, 0, 2, 3)

    def __len__(self):
         return len(self.files_path)
    
    def set_split(self, split):
        self.split = split
        print("Current split", self.split)
    
    def __get_categorical_label(self, idx):
        if self.classtype=="12emotions":
            dict = emo_fake_true_12
        elif self.classtype=="2emotions":
            dict = emofake_true
        elif self.classtype=="6emotions":
            dict = emo_basic_6
        else:
            raise Exception("not allowed")
        label = dict[self.labels[idx]]
        return label
    
    