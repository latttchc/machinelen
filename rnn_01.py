import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch import optim


sinx = torch.linspace(-2*math.pi, 2*math.pi, 100)
#ノイズを乗せる
siny = torch.sin(sinx)+ 0.1*torch.randn(len(sinx))

n_time = 10
n_sample = len(sinx) - n_time

input_data = torch.zeros((n_sample, n_time, 1))
correct_data = torch.zeros((n_sample, 1))

for i in range(n_sample):
    input_data[i]= siny[i:i+n_time].view(-1,1)
    correct_data[i] = siny[i+n_time:i+n_time+1]

dataset = TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset,batch_size=8,shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(input_size=1,
                          hidden_size=64,
                          batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self,x):
        y_rnn , h = self.rnn(x, None)
        y = self.fc(y_rnn[:,-1,:])#batch_firstの入力形状

        return y

net = Net()

loss_func = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01)
record_loss_train = []

epochs = 100
for i in range(100):
    net.train()
    loss_train = 0
    for j, (x,t) in enumerate(train_loader):
        y = net(x)
        loss = loss_func(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    if i % 10 == 0 or i == epochs - 1: 
        net.eval()
        print("Epochs:",i,"loss:",loss_train)
        predicted = list(input_data[0].view(-1))
        for i in range(n_sample):
            x = torch.tensor(predicted[-n_time:])
            x = x.view(1, n_time, 1)
            y = net(x)
            predicted.append(y[0].item())
        plt.plot(range(len(sinx)),siny,label="corecct")
        plt.plot(range(len(predicted)), predicted, label= "predicted")
        plt.legend()#二つのグラフを重ねる
        plt.show()