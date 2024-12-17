import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt


# GPU確認　使えるならGPUに自動設定する
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#データの設定 MNISTを用いる
batch_size = 64
train_set = datasets.MNIST('data/',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)

#識別器（偽物を判定）
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        #28×28ピクセル×1chなので入力は784次元
        #中間層は適当に512次元に
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        #活性化関数の設定 ReLU関数の改良版を設定
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)

#生成器（偽物を生成）
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        #第1層　入力128次元　出力1024次元　←適当
        self.fc1 = nn.Linear(128, 1024)
        #第2層　入力1024次元　出力2048次元　
        self.fc2 = nn.Linear(1024, 2048)
        #第3層　出力層　出力は28×28×1＝784次元とする
        self.fc3 = nn.Linear(2048, 784)
        #活性化関数の設定
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        #最終出力は1ch 28×28とする
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)

#ネットワークのインスタンス（識別器と生成器を作成）
#インスタンス後、GPUもしくはCPUに配置
G = generator().to(device)
D = discriminator().to(device)


#損失関数の設定
loss = nn.BCELoss()

#それぞれのネットワークの最適化アルゴリズムを設定
lr = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))



#学習　
#生成器は見破られない偽物を作れるようになることを目指す
#識別機は精巧な偽物も見破れるようになることを目指す
epochs = 50
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        i += 1

        #識別器の学習
        #本物の入力は，MNISTデータセットの実際の画像
        real_inputs = imgs.to(device)
        #本物の判定結果を出力
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        #ランダムノイズから偽物データを生成する
        noise = (torch.rand(real_inputs.shape[0],128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        #偽物の判定結果を出力
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        #本物と偽物を出力結果を比較させてパラメータを更新させる
        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()


        #生成器の学習
        #生成器は識別者に本物と思わせるように学習させる
        #偽物データを生成
        noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        #作った偽物を識別器に渡して判定結果を出力させる（だませたか確認）
        fake_outputs = D(fake_inputs)
        #作った偽物の答え（だましたい数字の情報）
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        #だませたかどうかの結果と、だましたい数字を比較させてパラメータを更新させる
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if i % 100 == 0 or i == len(train_loader):
            print("Epoch：",epoch)
            print("discriminator_loss:",D_loss.item())
            print("generator_loss:",G_loss.item())
            
    #エポック10回毎に生成した偽物を表示
    if epoch % 10 == 0 or epoch == epochs-1:
        noise = torch.rand(28,128)
        noise = noise.to(device)
        fake_inputs = G(noise)
        print("fake")
        for i in range(10):
            ax = plt.subplot(1,10,i+1)
            ax.imshow(fake_inputs[i].detach().numpy().reshape(28,28))
        plt.show()
