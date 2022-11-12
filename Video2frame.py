import os
import sys
sys.path.insert(0, "../")
import numpy as np
import cv2
from pathlib import Path
# import dlib
from tqdm import tqdm
import json
from facenet_pytorch import MTCNN
import torch
import re
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class Preparator():
    def __init__(self):
        self.skipped_frames={}


# def face_detector_dlib(images):
    
#     ts, w, h, c = images.shape

#     aligned_faces = []

#     #for every frame, detect the face and align it
#     for t in tqdm(range(ts)):
        
#         img = images[t]

#         img[:,:,0] = dlib.equalize_histogram(img[:, :, 0])
#         img[:,:,1] = dlib.equalize_histogram(img[:, :, 1])
#         img[:,:,2] = dlib.equalize_histogram(img[:, :, 2])

#         rects = detector(img, 1)

#         detected_face =(predictor(img, rects[0]))
#         afaces = dlib.get_face_chip(img, detected_face, size=256)

#         aligned_faces.append(afaces)

#     #return the aligned faces of the current subject
#     return aligned_faces

    def face_detector_mtcnn(self,filepath,subject, image_size=224 ):
        
        if torch.cuda.is_available():
                device='cuda'
        else:
            device='cpu'
        mtcnn = MTCNN(image_size=image_size, margin=40, keep_all=True, select_largest=False, post_process=False, device=device)
        cap = cv2.VideoCapture(filepath)
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        np_frames = []
        skipped = []
        # Loop through video and detect the faces
        
        for i in tqdm(range(v_len)):
            ret, frame = cap.read()
            # cv2.imshow("frame", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if ret == False:
                skipped.append("frame %d" % i)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)
            np_frames.append(face[0].permute(1, 2, 0).int().numpy())
        key = str(subject) + Path(filepath).parts[-1]
        if len(skipped)>0:
            self.skipped_frames[key] = "skipped frames #" + str(len(skipped))
        cap.release()
        # print("empty frame found", skipped)
        # print(np.array(np_frames).shape)
        return np.array(np_frames)

    def get_indices(self,data, K):
        #initial shape -> (# of frames, width, height,channels)
        data = data.reshape((data.shape[0], -1)) # reshape as (# of frames,(w*h*c))
        
        #to exclude the initial third of the video (avoid neutral emotions)
        third = data.shape[0] // 3 

        #apply k means and obtain meaningful frame idxs
        kmeans= KMeans(n_clusters=K)
        kmeans.fit(data[third:])
        centers = kmeans.cluster_centers_
        dists = pairwise_distances(data[third:], centers)
        indices_1 = np.sort(np.argmin(dists, axis=0))

        indices = indices_1 + third
        return indices