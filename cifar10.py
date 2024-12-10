from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim

#CIFAR10データのダウンロード
cifar10_train = CIFAR10("./data",
                       train = True,
                       download = True,
                       transform = transforms.ToTensor())

cifar10_test = CIFAR10("./data",
                       train = False,
                       download = True,
                       transform = transforms.ToTensor())

#CIFAR10の画像分類クラス
cifar10_class = ["airplane",
                 "automobile",
                 "bird",
                 "cat",
                 "deer",
                 "dog",
                 "frog",
                 "horse",
                 "ship",
                 "truck"]

print("データの数:",len(cifar10_train), len(cifar10_test))


train_loader = DataLoader(cifar10_train,
                          batch_size = 25,
                          shuffle = True)

test_loader = DataLoader(cifar10_test,
                         batch_size = 25,
                         shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #畳み込み層1の定義（3ch,カーネル数8,カーネルサイズ5）
        self.conv1 = nn.Conv2d(3,8,5)
        #畳み込み層2の定義（6ch,カーネル数16,カーネルサイズ5）
        self.conv2 = nn.Conv2d(8,16,5)
        #プーリング層の定義(領域サイズ2,ストライド2)
        self.pool = nn.MaxPool2d(2,2)
        #ドロップアウトの設定
        self.dropout = nn.Dropout(p=0.5)
        #活性化関数の設定
        self.relu = nn.ReLU() 
        #全結合層の設定
        self.fc1 = nn.Linear(16*5*5,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1,16*5*5) #入力データを1次元に変換
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
#ニューラルネットワークをインスタンス
net = Net()
#もしGPUを搭載していてCUDAが使えるなら#を外して下記を実行する
#net.cuda()

#損失関数として交差エントロピー関数を定義する
loss_func = nn.CrossEntropyLoss()

#最適化手法としてAdamを用いる
optimizer = optim.Adam(net.parameters())

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

#グラフの作成
plt.plot(range(len(record_loss_train)),#横軸の設定
         record_loss_train, #縦軸の値の設定
         label = "loss_train") #グラフのラベル付け

plt.plot(range(len(record_loss_test)),#横軸の設定
         record_loss_test, #縦軸の値の設定
         label = "loss_test") #グラフのラベル付け

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss_func_value")
plt.show()

cifar10_loader = DataLoader(cifar10_test,
                            batch_size=1,
                            shuffle=True)

dataiter = iter(cifar10_loader)
images, labels = dataiter.__next__()

plt.imshow(images[0].permute(1,2,0))
plt.show()

net.eval()
y = net(images[0])
print("正解:", cifar10_class[labels[0]],
      "分類結果:",cifar10_class[y. argmax().item()])
