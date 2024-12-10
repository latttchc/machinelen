import torch
from torch import nn
import matplotlib.pyplot as plt
from torch import optim

#Scikit-learnのデータセットモジュール
from sklearn import datasets
#Scikit-learnのデータを学習用とテスト用に分割するモジュール
from sklearn.model_selection import train_test_split

#手書きの数字画像セットをダウンロード
digits_data = datasets.load_digits()

#画像セットをdigits_imagesに入力
digits_images = digits_data.data
#正解ラベルをlabelsに入力
labels = digits_data.target

#データを学習用と評価用に分割
x_train, x_test, t_train, t_test = train_test_split(digits_images,labels)

#tensorに変換
x_train = torch.tensor(x_train,dtype=torch.float32)
t_train = torch.tensor(t_train,dtype=torch.int64)
x_test = torch.tensor(x_test,dtype=torch.float32)
t_test = torch.tensor(t_test,dtype=torch.int64)

#ネットワークの構築
net = nn.Sequential(
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10))

#学習
#損失関数の定義
loss_func = nn.CrossEntropyLoss()

#SGDでパラメータ探索
optimizer = optim.SGD(net.parameters(),lr=0.01)

#ログ
record_loss_train = []
record_loss_test = []

for i in range(1000):
    #パラメータ勾配の初期化
    optimizer.zero_grad()

    #まず適当に出力する（順伝播）
    y_train = net(x_train)
    y_test = net(x_test)

    #学習時のログを記録する
    loss_train = loss_func(y_train, t_train)
    loss_test = loss_func(y_test, t_test)
    record_loss_train.append(loss_train.item())
    record_loss_test.append(loss_test.item())

    #逆伝播（勾配を計算）
    loss_train.backward()

    #パラメータの更新
    optimizer.step()

    #100回学習毎に損失関数の値を表示
    if i % 100 == 0:
        print("Epoch:",i,
              "Loss_train:",loss_train.item(),
              "Loss_test:",loss_test.item())
        




