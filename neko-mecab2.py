import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

f = open("wakati.txt", "r")

word2id = {}
id2word = {}

text = f.readlines()
num = len(text)
id = 0 
for i in range(num):
    for word in text[i].split(" "): #split:List内の要素を分割
        if word == "\n":
            continue
        else:
            if word not in word2id:
                id = id + 1
                word2id[word] = id #word to id
                id2word[id] = word #id to word
f.close()
print("word2id:",word2id)
print("id2word:",id2word)

def w2id(word2id, length):
    result = []
    f = open("wakati.txt", "r")
    text = f.readlines()
    for i in range(len(text)):
        tmp = []
        for word in text[i].split(" "):
            if word in word2id:
                tmp.append(word2id[word])
            if word == "\n":
                num = len(tmp)
                if num < length:
                    for i in range(length - num):
                        tmp.append(0)
                result.append(tmp)
    f.close()
    return result

id_list = w2id(word2id, 30)
print("id_list:",id_list)

def id2w(id2word, id_list):
    result = []
    for line in id_list:
        tmp = []
        for i in line:
            if i != 0:
                tmp.append(id2word[i])
        result.append(tmp)
    return result

dst = id2w(id2word, id_list)
print(dst)

class MyDataset(Dataset):
    def __init__(self, id_list):
        super().__init__
        self.x = [data[0:-1] for data in id_list]
        self.y = [data[1:] for data in id_list]
        self.data_length = len(id_list)
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self,idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
dataset = MyDataset(id_list)
print("dataset[0]:",dataset[0])