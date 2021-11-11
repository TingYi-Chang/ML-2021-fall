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
import collections

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


"""#### Set arguments and random seed"""

TRA_PATH = 'data/train/'
LABEL_PATH = 'data/train.csv'
MODEL_PATH = 'model.pth'
VAL_PATH = 'val.csv'
DEVICE_ID = 2
SEED = 1
NUM_ECPOCH = 1000

#torch.cuda.set_device(DEVICE_ID)
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
    train_image = [f'data/train/{i+7000}.jpg' for i in range(len(train_label))]
    for i in range(10):
        print (train_image[i], train_label[i])
    print (collections.Counter(train_label))
    print (len(train_label))
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    split_len = int(len(train_data) * valid_ratio)
    train_set = train_data[split_len:]
    valid_set = train_data[:split_len]
    return train_set, valid_set

def compute_statistics(dataset):
    data = []
    for (img_path, label) in dataset:
        data.append(np.array(Image.open(img_path)))
    data = np.array(data)
    return data.mean(), data.std()

train_set, valid_set = load_train_data(TRA_PATH, LABEL_PATH)
