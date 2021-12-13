# -*- coding: utf-8 -*-
"""hw4_sample_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dJIB6Sbd_T_S7HsP0pDfLQKYaxnARG1g
"""

import torch
import os
import csv
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


TRAIN_PATH = "data/train_label.csv"
TEST_PATH = "data/test.csv"
PRED_PATH = "sample_early_stopping_pred.csv"

BATCH_SIZE = 128
EPOCH_NUM = 30
MAX_POSITIONS_LEN = 500
SEED = 1
MODEL_DIR = 'sample_early_stopping_model.pth'
lr = 0.001

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w2v_config = {'path': 'w2v_3iter.model', 'dim': 256}
lstm_config = {'hidden_dim': 256, 'num_layers': 2, 'bidirectional': True, 'fix_embedding': False}
header_config = {'dropout': 0.5, 'hidden_dim': 512}
assert header_config['hidden_dim'] == lstm_config['hidden_dim'] or header_config['hidden_dim'] == lstm_config['hidden_dim'] * 2

def parsing_text(text):
    # TODO: do data processing
    return text

def load_train_label(path):
    tra_lb_pd = pd.read_csv(path)
    label = torch.FloatTensor(tra_lb_pd['label'].values)
    idx = tra_lb_pd['id'].tolist()
    text = [parsing_text(s).split(' ') for s in tra_lb_pd['text'].tolist()]
    return idx, text, label

def load_train_nolabel(path):
    tra_nlb_pd = pd.read_csv(path)
    text = [parsing_text(s).split(' ') for s in tra_nlb_pd['text'].tolist()]
    return text

def load_test(path):
    tst_pd = pd.read_csv(path)
    idx = tst_pd['id'].tolist()
    text = [parsing_text(s).split(' ') for s in tst_pd['text'].tolist()]
    return idx, text

class Preprocessor:
    def __init__(self, sentences, w2v_config):
        self.sentences = sentences
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        self.build_word2vec(sentences, **w2v_config)
        
    def build_word2vec(self, x, path, dim):
        if os.path.isfile(path):
            #print("loading word2vec model ...")
            w2v_model = Word2Vec.load(path)
        else:
            #print("training word2vec model ...")
            w2v_model = Word2Vec(x, size=dim, window=5, min_count=2, workers=12, iter=2, sg=1)
            #print("saving word2vec model ...")
            w2v_model.save(path)
            
        self.embedding_dim = w2v_model.vector_size
        for i, word in enumerate(w2v_model.wv.vocab):
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(w2v_model[word])
        
        self.embedding_matrix = torch.tensor(np.array(self.embedding_matrix))
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        #print("total words: {}".format(len(self.embedding_matrix)))
        
    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)   
        
    def sentence2idx(self, sentence):
        sentence_idx = []
        for word in sentence:
            if word in self.word2idx.keys():
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return torch.LongTensor(sentence_idx)
    
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, sentences, labels, preprocessor):
        self.id_list = id_list
        self.sentences = sentences
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __getitem__(self, idx):
        if self.labels is None: return self.id_list[idx], self.preprocessor.sentence2idx(self.sentences[idx])
        return self.id_list[idx], self.preprocessor.sentence2idx(self.sentences[idx]), self.labels[idx]
    
    def __len__(self):
        return len(self.sentences)
    
    def collate_fn(self, data):
        id_list = torch.LongTensor([d[0] for d in data])
        lengths = torch.LongTensor([len(d[1]) for d in data])
        texts = pad_sequence(
            [d[1] for d in data], batch_first=True).contiguous()
     
        if self.labels is None: 
            return id_list, lengths, texts
        
        labels = torch.FloatTensor([d[2] for d in data])
        return id_list, lengths, texts, labels

train_idx, train_label_text, label = load_train_label(TRAIN_PATH)
preprocessor = Preprocessor(train_label_text, w2v_config)

train_idx, valid_idx, train_label_text, valid_label_text, train_label, valid_label = train_test_split(train_idx, train_label_text, label, test_size=0.12)
train_dataset, valid_dataset = TwitterDataset(train_idx, train_label_text, train_label, preprocessor), TwitterDataset(valid_idx, valid_label_text, valid_label, preprocessor)

test_idx, test_text = load_test(TEST_PATH)
test_dataset = TwitterDataset(test_idx, test_text, None, preprocessor)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True,
                                            collate_fn = train_dataset.collate_fn,
                                            num_workers = 8)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False,
                                            collate_fn = valid_dataset.collate_fn,
                                            num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False,
                                            collate_fn = test_dataset.collate_fn,
                                            num_workers = 8)

class LSTM_Backbone(torch.nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, bidirectional, fix_embedding=True):
        super(LSTM_Backbone, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        
        self.lstm = torch.nn.LSTM(embedding.size(1), hidden_dim, num_layers=num_layers, \
                                  bidirectional=bidirectional, batch_first=True)
        
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs)
        return x
    
class Header(torch.nn.Module):
    def __init__(self, dropout, hidden_dim):
        super(Header, self).__init__()
        # TODO: you should design your classifier module
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                         torch.nn.Linear(hidden_dim, 1),
                                         torch.nn.Sigmoid())
        
    @ torch.no_grad()
    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = torch.arange(MAX_POSITIONS_LEN)[:lengths.max().item()].unsqueeze(
            0).expand(len(lengths), -1).to(lengths.device)
        length_masks = (ascending < lengths.unsqueeze(-1)).unsqueeze(-1)
        return length_masks
    
    def forward(self, inputs, lengths):
        # the input shape should be (N, L, D∗H)
        pad_mask = self._get_length_masks(lengths)
        inputs = inputs * pad_mask
        inputs = inputs.sum(dim=1)
        out = self.classifier(inputs).squeeze()
        return out

def train(train_loader, backbone, header, optimizer, criterion, device, epoch):

    total_loss = []
    total_acc = []
    
    for i, (idx_list, lengths, texts, labels) in enumerate(train_loader):
        lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if not backbone is None:
            inputs = backbone(inputs)
        soft_predicted = header(inputs, lengths)
        loss = criterion(soft_predicted, labels)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            hard_predicted = (soft_predicted >= 0.5).int()
            correct = sum(hard_predicted == labels).item()
            batch_size = len(labels)
        
        print('[ Epoch {}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, len(train_loader), loss.item(), correct * 100 / batch_size), end='\r')
    return (loss.item(), correct * 100 / batch_size)

def valid(valid_loader, backbone, header, criterion, device, epoch):
    backbone.eval()
    header.eval()
    with torch.no_grad():
        total_loss = []
        total_acc = []
        
        for i, (idx_list, lengths, texts, labels) in enumerate(valid_loader):
            lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)

            if not backbone is None:
                inputs = backbone(inputs)
            soft_predicted = header(inputs, lengths)
            loss = criterion(soft_predicted, labels)
            total_loss.append(loss.item())
            
            hard_predicted = (soft_predicted >= 0.5).int()
            correct = sum(hard_predicted == labels).item()
            acc = correct * 100 / len(labels)
            total_acc.append(acc)
            
            print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, np.mean(total_loss), np.mean(total_acc)), end='\r')
    backbone.train()
    header.train()
    return np.mean(total_loss), np.mean(total_acc)
       
def run_training(train_loader, valid_loader, backbone, header, epoch_num, lr, device, model_dir): 
    def check_point(backbone, header, loss, acc, best_acc, no_improve_time, model_dir):
        # TODO
        if (acc > best_acc):
            best_acc = acc
            no_improve_time = 0
            torch.save({'backbone': backbone.state_dict(), 'header': header.state_dict()}, model_dir)
        else:
            no_improve_time = no_improve_time+1
        return best_acc, no_improve_time

    def is_stop(loss, acc, best_acc, no_improve_time):
        # TODO
        if (no_improve_time>3):
            return True
        return False
    
    if backbone is None:
        trainable_paras = header.parameters()
    else:
        trainable_paras = list(backbone.parameters()) + list(header.parameters())
        
    optimizer = torch.optim.Adam(trainable_paras, lr=lr)
    
    backbone.train()
    header.train()
    backbone = backbone.to(device)
    header = header.to(device)
    criterion = torch.nn.BCELoss()
    best_acc = 0
    no_improve_time = 0
    for epoch in range(epoch_num):
        train_loss, train_acc = train(train_loader, backbone, header, optimizer, criterion, device, epoch)
        print('[Train in epoch {:}] loss:{:.3f} acc:{:.3f} '.format(epoch+1, train_loss, train_acc))

        loss, acc = valid(valid_loader, backbone, header, criterion, device, epoch)
        print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f} '.format(epoch+1, loss, acc))
        best_acc, no_improve_time = check_point(backbone, header, loss, acc, best_acc, no_improve_time, model_dir)
        print ("No improvement time: {:}, best acc:{:}".format(no_improve_time, best_acc))
        if is_stop(loss, acc, best_acc, no_improve_time):
            break

backbone = LSTM_Backbone(preprocessor.embedding_matrix, **lstm_config)
header = Header(**header_config)

if __name__ == '__main__':
    run_training(train_loader, valid_loader, backbone, header, EPOCH_NUM, lr, device, MODEL_DIR)
    #run_testing(test_loader, backbone, header, device, PRED_PATH)