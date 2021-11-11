import os
import random
import glob
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean


TST_PATH = '../../hw3/data/test/'
MODEL_NAME = 'best_model.pth'
PRED_NAME = 'pred.csv'
SEED = 1

#torch.cuda.set_device(DEVICE_ID)
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

def load_test_data(img_path):
    test_set = [f'{img_path}/{i}.jpg' for i in range(7000)]
    return test_set
    
test_set = load_test_data(TST_PATH)
   
class TestingDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        # TODO
        #print (type(data))
        return data
    
    def read_img(self, idx):
        img = Image.open(self.data[idx])
        if not self.augment is None:
            img = self.augment(img)
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).float()
        img = self.normalize(img)
        return img
        
    def __getitem__(self, idx):
        img = self.read_img(idx)
        return img

test_dataset = TestingDataset(test_set)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class FaceExpressionNet(nn.Module):
    def __init__(self):
        super(FaceExpressionNet, self).__init__()
        # TODO
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),   
            nn.Dropout2d(p=0.5),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, affine=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        #image size (64,64)
        x = self.conv(x) #(32,32)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

def test(test_loader, model):
    with torch.no_grad():
        predict_result = []
        for idx, img in enumerate(test_loader):
            if use_gpu:
                img = img.to(device)
            output = model(img)
            predict = torch.argmax(output, dim=-1).tolist()
            predict_result += predict
        
    with open(PRED_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in range(len(predict_result)):
            writer.writerow([str(i), str(predict_result[i])])

if __name__ == '__main__':
    model = FaceExpressionNet()
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()
    if use_gpu:
        model.to(device)
    test(test_loader, model)

