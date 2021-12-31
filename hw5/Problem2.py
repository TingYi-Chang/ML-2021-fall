# -*- coding: utf-8 -*-
"""ml-2021-fall-hw5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iGWE9Lx46gM3-Xm3837KYzw1EojwM9Mc

# Import Module
"""

import csv
import time
import sys
import os
import random

# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import matplotlib.pyplot as plt

"""# Set Hyper-parameters"""

BATCH_SIZE = 256
LATENT_DIM = 0
REDUCED_DIM = 2
NUM_ITER = 1000
REDUCED_METHOD = 'tsne'
MODEL_NAME = 'model_best_test.pth'
PRED_PATH = 'best_modify.csv'

DATA_PATH = 'visualization.npy'
DEVICE_ID = 0
SEED = 1040

torch.cuda.set_device(DEVICE_ID)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

"""# Define Dataset"""

class Dataset(data.Dataset):
    def __init__(self, data_path):
        self.total_img = torch.from_numpy(np.load(data_path)).float()
        self.total_img = self.total_img.permute(0, 3, 1, 2)
        
    def normalize(self, img):
        # TODO
        img = img / 255
        return img
    
    def augment(self, img):
        # TODO
        return img
    
    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        img = self.total_img[index]
        img_aug = self.augment(img)
        
        img_aug = self.normalize(img_aug)
        img = self.normalize(img)
        return img_aug, img

"""# Define Model Architerchure"""

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)
    
class Net(nn.Module):
    def __init__(self, image_channels=3, latent_dim=64):
        super(Net, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        
        # you should check the latent dimension (N)
        #self.fc1 = nn.Linear(128*256*2*2, self.latent_dim)
        #self.fc2 = nn.Linear(self.latent_dim, 128*256*2*2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False),
            nn.Tanh(),
        )
                
    def forward(self, x):
        x_encode = self.encoder(x)
        x_decode  = self.decoder(x_encode)
        #print ("Shape:",x.shape,"After encode shape:",x_encode.shape,", after decode shape:",x_decode.shape)
        return x_encode,x_decode
"""# Define Clustering Process"""

def clustering(model, device, loader, n_iter, reduced_method, reduced_dim):
    assert reduced_method in ['pca', 'tsne', None]
    
    model.eval()    
    latent_vec = torch.tensor([])#.to(device, dtype=torch.float)
    for idx, (image_aug, image) in enumerate(loader):
        print("predict %d / %d" % (idx, len(loader)) , end='\r')
        #image = image.to(device, dtype=torch.float)
        
        latent, r = model(image)
        print("latent shape:" ,latent.shape , end='\r')
        latent_vec = torch.cat((latent_vec, latent.view(image.size()[0], -1)), dim=0)
    
    latent_vec = latent_vec.cpu().detach().numpy()
    latent_vec = latent_vec.reshape((latent_vec.shape[0] , -1))
    print ("latent_vec before cluster:",latent_vec.shape)
    pca = PCA(n_components=100, copy=False, whiten=True, svd_solver='full')
    latent_vec = pca.fit_transform(latent_vec)
    print ("latent_vec after PCA:",latent_vec.shape)
    if reduced_method == 'tsne':
        tsne = TSNE()
        latent_vec = tsne.fit_transform(latent_vec)
    elif reduced_method == 'pca':
        pca = PCA(n_components=reduced_dim, copy=False, whiten=True, svd_solver='full')
        latent_vec = pca.fit_transform(latent_vec)
    print ("latent_vec after TSNE:",latent_vec.shape)
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(latent_vec)
    return kmeans.labels_

def reconstruct(model, loader):
    model.eval()
    with torch.no_grad():
        for idx, (image_aug, image) in enumerate(loader):
            
            image = np.transpose(image[1].cpu().numpy(), [1,2,0])
            plt.subplot(121)
            plt.imshow(np.squeeze(image))
            _, reconstruct_test = model(image_aug)
            outimg = np.transpose(reconstruct_test[1].cpu().numpy(), [1,2,0])
            plt.subplot(122)
            plt.imshow(np.squeeze(outimg))
            plt.show()
            break


"""# Define write function"""

def write_output(predict_result, file_name):      
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(len(predict_result)):
            writer.writerow([str(i), str(predict_result[i])])

"""# Main Process"""

# build dataset
dataset = Dataset(DATA_PATH)

if __name__ == '__main__':
    model = Net()#.to(device)
    
    test_loader = data.DataLoader(dataset, batch_size=128, shuffle=False)
    model.load_state_dict(torch.load(MODEL_NAME))
    reconstruct(model, test_loader)
    predicted = clustering(model, device, test_loader, NUM_ITER, reduced_method=REDUCED_METHOD, reduced_dim=REDUCED_DIM)
    print(predicted.shape)
    