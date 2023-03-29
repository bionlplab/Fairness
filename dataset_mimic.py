import pickle
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import time
import copy
import torchvision.transforms as transforms
import cv2
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, groups, transform=None):
        'Initialization'
        self.groups = groups
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
       # 'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
       # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        X = Image.open(ID).convert('RGB')
#         X = Image.open('/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/images/images/' + ID).convert('RGB')
        # X = cv2.imread('/prj0129/mil4012/glaucoma/NIH-chest-x-ray/CXR8/images/images/' + ID)
        # X = cv2.resize(X,(224,224))
        if self.transform:
             X = self.transform(X)
        
        y = self.labels[index]
        group = self.groups[index]
        

        # return X, torch.tensor(y),torch.tensor(group)
        # return X, torch.tensor(y),group
        return X, torch.FloatTensor(y),group