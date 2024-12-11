from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch

#データセットのセットアップ
train = FashionMNIST("./data", train=True,download=True,transform=transforms.ToTensor())
test = FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())

print("学習データ数:",len(train))
print("テストデータ数:",len(test))

fashion_class = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#バッチサイズ
Batch_size = 128

#DataLoader
train_loader = DataLoader(train,batch_size=Batch_size,shuffle=True)
test_loader =  DataLoader(test, batch_size=Batch_size,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #ニューラルネットワークのレイヤー
        #入力次元は28✖️28ピクセル
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,128)
        #出力次元は10次元
        self.fc4 = nn.Linear(128,10)

        #活性化関数
        self.relu = nn.ReLU()
    
    def forward(self,x):
        #画像を1次元にデータ変換
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    
#ニューラルネットワークをインスタンス
net = Net()
#損失関数(交差エントロピー関数)
loss_func = nn.CrossEntropyLoss()
#最適化手法としてAdamを用いる
optimizer = optim.Adam(net.parameters())

#損失関数の値
record_loss_train = []
record_loss_test = []

#エポック数:10
for i in range(10):
    #学習モードにする
    net.train()
    #損失関数の値
    loss_train = 0
    #ミニバッチ
    for j, (x,t) in enumerate(train_loader):
        #順伝播
        y = net(x)
        #損失関数を計算
        loss = loss_func(y, t)
        loss_train += loss.item()
        #勾配計算と逆伝播
        optimizer.zero_grad()
        loss.backward()
        #パラメータを更新
        optimizer.step()
    #損失関数の値
    loss_train /= j+1
    record_loss_train.append(loss_train)

    #netを評価モードにする
    net.eval()
    loss_test = 0
    for j, (x,t) in enumerate(test_loader):
        #順伝播
        y = net(x)
        #損失関数の計算
        loss = loss_func(y, t)
        loss_test += loss.item()
    #損失関数の値
    loss_test /= j+1
    record_loss_test.append(loss_test)
    #損失関数の値を表示
    print("Epoch:", i, "Loss_train:", loss_train, "Loss_test",loss_test)

#グラフの定義
plt.plot(range(len(record_loss_train)), record_loss_train, label="loss_train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="loss_test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


fashion_loader = DataLoader(test, batch_size=1, shuffle=True)
dataiter = iter(fashion_loader)
images, labels = next(dataiter)

#データを出力
plt.imshow(images[0].squeeze(), cmap="gray")
plt.show()

net.eval()
y = net(images)
print(f"正解: {fashion_class[labels[0].item()]}, 推論: {fashion_class[y.argmax().item()]}")