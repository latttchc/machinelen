from torchvision.datasets import MNIST
from torchvision import transforms

from torch.utils.data import DataLoader

import torch.nn as nn

from torch import optim

import matplotlib.pyplot as plt

#学習データのダウンロード
mnist_train = MNIST("./data",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor())

#テストデータのダウンロード
mnist_test = MNIST("./data",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor())

print("学習データの数:",len(mnist_train))
print("テストデータの数:",len(mnist_test))

#バッチサイズの設定
Batch_size = 256

#DataLoaderの設定
train_loader = DataLoader(mnist_train,
                          batch_size = Batch_size,
                          shuffle = True)

test_loader = DataLoader(mnist_test,
                          batch_size = Batch_size,
                          shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #ニューラルネットワークのレイヤーの定義
        #入力次元は手書き数字画像のサイズ28×28ピクセル
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128,256)
        #出力次元は0-9の10次元
        self.fc3 = nn.Linear(256,10)
        
        #活性化関数の定義
        self.relu = nn.ReLU()
        
    def forward(self,x):
        #画像を1次元のデータに変換
        x = x.view(-1, 28*28)
        #ネットワークを構築
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
         
        return x

#ニューラルネットワークをインスタンス
net = Net()
#もしGPUを搭載していてCUDAが使えるなら#を外して下記を実行する
#net.cuda()

#損失関数として交差エントロピー関数を定義する
loss_func = nn.CrossEntropyLoss()

#最適化手法としてSGDを用いる学習率は0.01とする
optimizer = optim.SGD(net.parameters(), lr = 0.01)

#損失関数の値のログ
record_loss_train = []
record_loss_test = []

#学習 エポック数は10とする
for i in range(10):
    #ネットワークnetを学習モードにする
    net.train()
    #損失関数の値を定義
    loss_train = 0
    #ミニバッチ学習
    for j, (x, t) in enumerate(train_loader):
        #もしGPUを搭載していてCUDAが使えるなら下記の#を外してGPUにデータを配置
        #x, t = x.cuda(), t.cuda()
        #順伝播を計算
        y = net(x)
        #損失関数の計算
        loss = loss_func(y, t)
        loss_train += loss.item()
        #勾配計算と逆伝播
        optimizer.zero_grad()
        loss.backward()
        #パラメータを更新
        optimizer.step()
    #損失関数の値を記録(ミニバッチ学習毎に損失関数の値の平均を計算)
    loss_train /= j+1
    record_loss_train.append(loss_train)

    #ネットワークnetを評価モードにする
    net.eval()
    loss_test = 0
    #ミニバッチとしてテストデータを取り出して評価
    for j, (x, t) in enumerate(test_loader):
        #もしGPUを搭載していてCUDAが使えるなら下記の#を外してGPUにデータを配置
        #x, t = x.cuda(), t.cuda()
        #順伝播を計算
        y = net(x)
        #損失関数の計算
        loss = loss_func(y, t)
        loss_test += loss.item()
    #損失関数の値を記録(ミニバッチ毎に損失関数の値の平均を計算)
    loss_test /= j+1
    record_loss_test.append(loss_test)

    #損失関数の値を表示
    print("Epoch:", i, "Loss_train:", loss_train, "Loss_test:",loss_test)
plt.plot(range(len(record_loss_train)),
         record_loss_train,
         label = "loss_train")
plt.plot(range(len(record_loss_test)),
        label = "loss_test")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("loss_func_val")
plt.show() 


               

    
    



