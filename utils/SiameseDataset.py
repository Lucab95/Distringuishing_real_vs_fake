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
    
    

class SiameseNetworkDataset(Dataset):
    def __init__(self,video_folder, n_emt=12, required_frames = 1,transforms=None):
        self.video_folder = video_folder
        self.n_emt = n_emt
        self.files_path = []
        self.labels = []
        self.subject = []
        default_transforms = [ToTensor()]
        if transforms is not None:
            default_transforms += transforms
        self.transforms = Compose(default_transforms)
        self.K = 50
        self.get_file_paths()
        self.files_path= sorted(self.files_path, key=lambda i: int((re.split('([0-9]+)',i)[colab_idx])))
        self.get_file_labels()
        #skip offest frames to remove neutral part from start and end
        offset = 10

        self.indexes = np.arange(offset,self.K-offset,(self.K-offset*2)/(required_frames+1)).astype(int)[4]
        print(self.indexes)
        # self.indexes = np.arange(0,self.K,self.K /required_frames).astype(int)
        # self.indexes = np.arange(0,self.K,).astype(int)

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
            v_frames_0 = np.load(f)  # load all the frames having this shape: (T, H, W, C) T=frame number
        
        label = self.__get_categorical_label(idx)
        
        # .astype(np.float32)


        
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                
                idx_1 = random.randint(0,len(self.files_path)-1)
                if label == self.__get_categorical_label(idx_1):
                    break
        else:
            while True:
                #Look untill a different class image is found
                idx_1 = random.randint(0,len(self.files_path)-1)
                if label == self.__get_categorical_label(idx_1):
                    break
        
        with open(self.files_path[idx_1], "rb") as f:
            v_frames_1 = np.load(f)

        # get the necessary indexes of frames for the videos
        v_frames_0= v_frames_0[self.indexes].astype(np.uint8)
        v_frames_1= v_frames_1[self.indexes].astype(np.uint8)

        #apply the custom transformations
        if self.transforms:
            video_frames_0 = []
            video_frames_1 = []

            for i in range(len(v_frames_0)):
                video_frames_0.append(self.transforms((v_frames_0[i])))
                video_frames_1.append(self.transforms((v_frames_1[i])))

            video_frames_0 = torch.stack(video_frames_0, 0)
            video_frames_1 = torch.stack(video_frames_1, 0)

        
        

        v_frames = video_frames_0
        v_frames_1 = video_frames_1
        label = torch.tensor(label)

        # label = torch.from_numpy(np.array(([label != self.__get_categorical_label(idx_1)])))
        return v_frames, v_frames_1, label
        # res_vframes = res_vframes.permute(1, 0, 2, 3)


    def __get_categorical_label(self, idx):
        if self.n_emt==12:
            dict = emo_fake_true_12
        elif self.n_emt==2:
            dict = emofake_true
        elif self.n_emt==6:
            dict = emo_basic_6
        else:
            raise Exception("number of emotions not allowed")
        label = dict[self.labels[idx]]
        return label