import os
import random
import glob
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


TRA_PATH = '../../hw3/data/train/'
LABEL_PATH = '../../hw3/data/train.csv'
MODEL_PATH = 'model.pth'
VAL_PATH = 'val.csv'
NUM_ECPOCH = 1000

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

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

class FaceExpressionDataset(Dataset):
    def __init__(self, data, augment=None):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)
    
    def normalize(self, data):
        
        #mean = [0.5, 0.5, 0.5]
        #std = [0.1, 0.1, 0.1]
        #transforms.Normalize(mean, std)
        # TODO
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
 
valid_dataset = FaceExpressionDataset(valid_set)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

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

def valid_save(valid_loader, model, loss_fn, use_gpu=True, file_name=VAL_PATH):
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        predict_result = []

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
            predict = torch.argmax(output, dim=-1).tolist()
            predict_result += predict
       
        valid_acc = np.mean(valid_acc)
        valid_loss = np.mean(valid_loss)
        print("valid Loss: {:.4f}, valid Acc: {:.4f}".format(valid_loss, valid_acc))

        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'label'])
            for i in range(len(predict_result)):
                writer.writerow([str(i), str(predict_result[i])])

def plot_confusion_matrix(y_hat, y_pred, title):
    confmat = confusion_matrix(y_hat, y_pred)

    classes = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    classes = classes[unique_labels(y_hat, y_pred)]
    confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(confmat, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confmat.shape[1]),
           yticks=np.arange(confmat.shape[0]),
           xticklabels=classes, 
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, format(confmat[i, j], '.2f'), ha="center", va="center") 
    
    return 

if __name__ == '__main__':
    model = FaceExpressionNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    if use_gpu:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    valid_save(valid_loader, model, loss_fn, use_gpu=True)
    y_hat = np.array([y[1] for y in valid_set])

    valid_data = pd.read_csv(VAL_PATH)
    y_pred = valid_data['label'].values
    
    plot_confusion_matrix(y_hat, y_pred, 'Confusion matrix')
    plt.savefig("confusion_matrix_1.png")