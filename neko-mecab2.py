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
    #単語毎にリスト内に分割
    for word in text[i].split(" "): #split:List内の要素を分割
        if word == "\n": #　\nは無視
            continue

        else: 
            if word not in word2id: #辞書内に単語が存在しなければidを割り当てて割り当てて追加
                id = id + 1
                word2id[word] = id #word to id
                id2word[id] = word #id to word
f.close()
print("word2id:",word2id)
print("id2word:",id2word)

#分かち書きされた文章からIDリストにを作成
def w2id(word2id, length):
    result = [] #結果を格納
    f = open("wakati.txt", "r")
    text = f.readlines()
    for i in range(len(text)):
        tmp = []
        for word in text[i].split(" "):
            if word in word2id:
                tmp.append(word2id[word]) #
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

BS = 2 #batch_size
dl = DataLoader(dataset,
                batch_size=BS,
                shuffle=True,
                drop_last=True)
iterator = iter(dl)
x_train, y_train = next(iterator)

VS = len(word2id)+1
ED = 5
embedding = nn.Embedding(VS,ED,padding_idx=0)
x = embedding(x_train)

print(x_train.shape)
print(x_train)
print(x)
print(y_train.shape)
print(y_train)



class Net(nn.Module):
    def __init__(self, VS, ED,HS,BS,NL=1):
        super().__init__()
        self.hidden_size = HS
        self.batch_size = BS
        self.num_layers = NL
        self.device = torch.device('cpu')
        self = self.to(self.device)

        self.emedding = nn.Embedding(VS, ED, padding_idx= 0)
        self.rnn = nn.RNN(ED,
                          HS, 
                          batch_first=True, 
                          num_layers=self.num_layers)
        self.fc = nn.Linear(HS,VS)
    
    def init_hidden(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        self.hidden_state=torch.zeros(self.num_layers,
                                      batch_size,
                                      self.hidden_size).to(self.device)
        
    def forward(self,x):
        x = self.emedding(x)
        x, self.hidden_state = self.rnn(x,self.hidden_state)
        x = self.fc(x)

        return x

HS = 200
NL = 1
model = Net(VS,ED,HS,BS,NL)
loss_func = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

Epochs = 1000
device = model.device
model.train()
losses = []

for epoch in range(Epochs):
    running_loss = 0
    for cnt, (X_train, Y_train) in enumerate(dl):
        optimizer.zero_grad()
        X_train, y_train = X_train.to(device), y_train.to(device)
        model.init_hidden()
        outputs = model(X_train)
        outputs = outputs.reshape(-1, VS)
        y_train = y_train.reshape(-1)
        loss = loss_func(outputs, y_train)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(running_loss/cnt)

    if epoch % 5 == 0:
        print(f'¥nepoch: {epoch:3}, loss:{loss:3f}')


plt.plot(losses)
plt.show()

device = model.device
model.eval()

words = ['吾輩']
word = words[0]
sentence = word

l = len(word2id)+1
id = [word2id[word]]
# tmp = torch.tensor(id).to(device)
tmp = torch.tensor(id).unsqueeze(0).to(device)
model.init_hidden(batch_size=1)
o = model(tmp)
print(o.shape)
o=o.reshape(l) #1次元ベクトルに変換
print(o.shape)

#上記の結果を確率に変換
probs = torch.nn.functional.softmax(o, dim=0).cpu().detach().numpy()
next_idx = np.random.choice(l,p=probs)
next_word = id2word[next_idx]
sentence += ' ' + next_word
print(sentence)

id = [[word2id[next_word]]]

def make_sentence(words,model,device, word2id, id2word):
    model.eval()
    batch_size = 1
    l = len(word2id)+1
    eos = ["。"]
    result = []
    with torch.no_grad():
        for word in words:
            model.init_hidden(batch_size)
            sentence = word
            id = [[word2id[word]]]
            for idx in range(50):
                t = torch.tensor(id).to(device)
                outputs = model(t)
                outputs = outputs.reshape(l)
                probs = torch.softmax(outputs, dim=0).cpu().detach().numpy()
                next_idx = np.random.choice(l,p=probs)
                next_word = id2word[next_idx]
                sentence += ' ' + next_word
                if next_word in eos:
                    break
                id = [[word2id[next_word]]]
            result.append(sentence.replace(" ",""))
    return result

print('-----------')
words = ["吾輩","名前","どこ","何","しかも","書生","しかし","ただ"]
result = make_sentence(words, model, device, word2id, id2word)
print(result)