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
    
    

class TripletNetworkDataset(Dataset):

    def __init__(self,video_folder, n_emt=12, required_frames = 1,transforms=None, test_subj=np.arange(0, 50, 1, dtype=int), train=False):
        random_state = np.random.RandomState(42)
        self.video_folder = video_folder
        self.n_emt = n_emt
        self.files_path = []
        self.labels = []
        self.subject = []
        self.required_frames = required_frames
        default_transforms = [ToTensor()]
        if transforms is not None:
            default_transforms += transforms
        self.transforms = Compose(default_transforms)
        self.K = 50
        self.train=train
        self.get_file_paths(test_subj)
        self.files_path= np.array(sorted(self.files_path, key=lambda i: int((re.split('([0-9]+)',i)[colab_idx]))))
        self.get_file_labels()
        #skip offest frames to remove neutral part from start and end
        offset = 10
        if required_frames==1:
            self.indexes = [40]
        else:
            self.indexes = np.arange(offset,self.K-offset,(self.K-offset*2)/(required_frames+1)).astype(int)[1:]
        # self.indexes = np.arange(0,self.K,self.K /required_frames).astype(int)
        # self.indexes = np.arange(0,self.K,).astype(int)


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (v_frames, label) where label is the emotion of the video
        """
        

        with open(self.files_path[idx], "rb") as f:
            anchor = np.load(f)  # load all the frames having this shape: (T, H, W, C) T=frame number


        anchor_label = self.labels[idx]
        anchor_subj = self.subject[idx]
        
        pos_idxs = np.where((self.labels == anchor_label)&(self.subject!=anchor_subj))[0]
        neg_idxs = np.where((self.labels != anchor_label)&(self.subject!=anchor_subj))[0]
        idx_pos = np.random.choice(pos_idxs)
        idx_neg = np.random.choice(neg_idxs)

        with open(self.files_path[idx_pos], "rb") as f:
            positive = np.load(f)
        with open(self.files_path[idx_neg], "rb") as f:
            negative = np.load(f)

        # get the necessary indexes of frames for the videos
        anchor = anchor[self.indexes].astype(np.uint8)
        positive= positive[self.indexes].astype(np.uint8)
        negative= negative[self.indexes].astype(np.uint8)

        #apply the custom transformations
        if self.transforms:
            video_frames_anch = []
            video_frames_pos = []
            video_frames_neg = []

            for i in range(len(anchor)):
                anchor_frame= self.transforms(anchor[i])
                positive_frame= self.transforms(positive[i])
                negative_frame= self.transforms(negative[i])
                

                # triplet.append((anchor_frame, positive_frame, negative_frame))
                video_frames_anch.append(anchor_frame)
                video_frames_pos.append(positive_frame)
                video_frames_neg.append(negative_frame)

            if self.required_frames==1:
                
                return anchor_frame , positive_frame , negative_frame
            
            video_frames_anch = torch.stack(video_frames_anch,0)
            # triplet = torch.stack(triplet,0)

        # label = torch.from_numpy(np.array(([label != self.__get_categorical_label(idx_1)])))
        # print(self.files_path[idx])
        # print(self.files_path[idx_pos])
        # print(self.files_path[idx_neg])
        
        # return triplet, anchor_label, self.labels[idx_neg]
        return video_frames_anch, anchor_label, self.labels[idx_neg]
        
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
        if isinstance(idx, str):
            numerical_label = dict[idx]
        else:
            numerical_label = dict[self.labels[idx]]
        return numerical_label

    def get_file_paths(self, test_subj): 
    # split to train / val
        format = "k"+ str(self.K)
        for root, d, files in os.walk(self.video_folder):
            for f in files:
                if f.endswith(format):
                    fpath = os.path.join(root, f)
                    if self.train:
                        #train case
                        if int(re.split('([0-9]+)',f)[1]) not in test_subj:
                            self.files_path.append(fpath)
                    else:
                        #test case
                        if int(re.split('([0-9]+)',f)[1]) in test_subj:
                            self.files_path.append(fpath)


                     

        self.files_path = self.files_path
                
        

    def get_file_labels(self):
        self.dog = np.array([Path(f).parts[-2] for f in self.files_path])
        for f in self.files_path:
            text_label=Path(f).parts[-2]
            self.labels.append(self.__get_categorical_label(text_label))
        self.text_labels = np.array([Path(f).parts[-2] for f in self.files_path])
        
        self.labels = np.array(self.labels)
        self.subject =  np.array([int(re.split('([0-9]+)',f)[colab_idx]) for f in self.files_path])
    def triplet_helper(self):
        return self.file_path, self.labels, self.subject

    def get_shape(self,idx):
        print("tensor shape:", self[idx][0].detach().cpu().numpy().shape, "\nlabel shape:", self[idx][1].detach().cpu().numpy().shape)

    def __len__(self):
        return len(self.files_path)