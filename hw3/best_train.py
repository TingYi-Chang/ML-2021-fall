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


"""#### Set arguments and random seed"""

TRA_PATH = '../../hw3/data/train/'
TST_PATH = '../../hw3/data/test/'
MODEL_PATH = 'model.pth'
LABEL_PATH = '../../hw3/data/train.csv'
SEED = 1
NUM_ECPOCH = 1000

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

"""#### Process data"""

def load_train_data(img_path, label_path, valid_ratio=0.12):
    train_label = pd.read_csv(label_path)['label'].values.tolist()
    train_image = [f'{TRA_PATH}{i+7000}.jpg' for i in range(len(train_label))]
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    split_len = int(len(train_data) * valid_ratio)
    train_set = train_data[split_len:]
    valid_set = train_data[:split_len]
    
    return train_set, valid_set

train_set, valid_set = load_train_data(TRA_PATH, LABEL_PATH)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(degrees=30),
])

"""#### Customize dataset"""

class FaceExpressionDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        return data
    
    def read_img(self, idx):
        img = Image.open(self.data[idx][0])
        if not self.augment is None:
            img = self.augment(img)
        img = torch.from_numpy(np.array(img)).float()
        img = img.unsqueeze(0).float()
        img = self.normalize(img)
        return img
    
    def __getitem__(self, idx):
        img = self.read_img(idx)
        label = self.data[idx][1]
        return img, label
    
class TestingDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
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

train_dataset = FaceExpressionDataset(train_set, transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = FaceExpressionDataset(valid_set)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

"""#### Define module class"""

class FaceExpressionNet(nn.Module):
    def __init__(self):
        super(FaceExpressionNet, self).__init__()
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

"""#### Define training and testing process"""

def train(train_loader, model, loss_fn, use_gpu=True):
    model.train()
    train_loss = []
    train_acc = []
    for (img, label) in train_loader:
        if use_gpu:
            img = img.to(device)
            label = label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()            
        optimizer.step()
        with torch.no_grad():
            predict = torch.argmax(output, dim=-1)
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
    print("Epoch: {}/{}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, NUM_ECPOCH, np.mean(train_loss), np.mean(train_acc)))
    return np.mean(train_acc)
    
def valid(valid_loader, model, loss_fn, use_gpu=True):
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        for idx, (img, label) in enumerate(valid_loader):
            if use_gpu:
                img = img.to(device)
                label = label.to(device)
            output = model(img)
            loss = loss_fn(output, label)
            predict = torch.argmax(output, dim=-1)
            acc = (label == predict).cpu().tolist()
            valid_loss.append(loss.item())
            valid_acc += acc
       
        valid_acc = np.mean(valid_acc)
        valid_loss = np.mean(valid_loss)
        print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))
    return valid_acc

def save_checkpoint(valid_acc, acc_record, epoch):
    if valid_acc >= np.mean(acc_record[-5:]):   
        torch.save(model.state_dict(), MODEL_PATH)
        print('model saved to %s' % MODEL_PATH)

if __name__ == '__main__':
    model = FaceExpressionNet()
    if use_gpu:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    train_record =[]
    valid_record = []
    
    for epoch in range(NUM_ECPOCH):
        train_acc = train(train_loader, model, loss_fn, use_gpu)
        valid_acc = valid(valid_loader, model, loss_fn, use_gpu=True)
        train_record.append(train_acc)
        valid_record.append(valid_acc)
        
        save_checkpoint(valid_acc, valid_record, epoch)
        #Early Stop
        """
        if valid_acc > best_acc:
            best_acc = valid_acc
            es = 0
        else:
            es += 1
            if es > patience:
                break
        """
        print('########################################################')
    torch.save(model.state_dict(), 'last_model.pth')
    x=np.arange(NUM_ECPOCH)
    plt.plot(x,train_record)
    plt.plot(x,valid_record)
    plt.savefig("5_best_acc.png")
