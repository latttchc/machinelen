import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#GPU確認　
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#データの設定MNISTを用いる
batch_size=64
train_set=datasets.MNIST('data/',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)

#生成器の設定
class generator(nn.Module):
    def __init__(self):
        super().__init__()
        #転置畳み込み演算層の定義
        #入力次元数,出力次元数,カーネルサイズ,ストライド数,パディングの値
        self.convt1=nn.ConvTranspose2d(100,512,3,1,0)
        self.convt2=nn.ConvTranspose2d(512,256,3,2,0)
        self.convt3=nn.ConvTranspose2d(256,128,4,2,1)
        self.convt4=nn.ConvTranspose2d(128,1,4,2,1)

        #バッチ正規化層
        self.bn2d1 = nn.BatchNorm2d(512)
        self.bn2d2 = nn.BatchNorm2d(256)
        self.bn2d3 = nn.BatchNorm2d(128)

        #活性化関数の設定
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.convt1(x))
        x = self.bn2d1(x)
        x = self.relu(self.convt2(x))
        x = self.bn2d2(x)
        x = self.relu(self.convt3(x))
        x = self.bn2d3(x)
        x = self.convt4(x)

        return nn.Tanh()(x)

#識別器の設定
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #畳み込み演算層の定義
        #入力次元数, 出力次元数, カーネルサイズ, ストライド数, パディングの値
        self.conv1 = nn.Conv2d(1,128,4,2,1)
        self.conv2 = nn.Conv2d(128,256,4,2,1)
        self.conv3 = nn.Conv2d(256,512,3,2,0)
        self.conv4 = nn.Conv2d(512,1,3,1,0)

        self.bn2d1 = nn.BatchNorm2d(128)
        self.bn2d2 = nn.BatchNorm2d(256)
        self.bn2d3 = nn.BatchNorm2d(512)

        #活性化関数
        self.lrelu = nn.LeakyReLU(0,2)
    
    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.bn2d2(x)
        x = self.lrelu(self.conv3(x))
        x = self.bn2d3(x)
        x = self.conv4(x)
        return nn.Sigmoid()(x)

#ネットワークのインスタンス(識別器と生成器を生成)
G = generator().to(device)
D = discriminator().to(device)

#損失関数の設定
loss = nn.BCELoss()

#損失関数の設定
lr = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

#学習
#生成器は見破られない偽物を作れるようになることを目指す
#識別器は精巧な偽物も見破れるようになることを目指す
epochs = 50
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        i += 1
        #バッチサイズ
        batch_size = imgs.size()[0]
        #ノイズ
        noise = torch.randn(batch_size, 100,1,1)

        #識別器の学習
        #本物の入力は,MNISTデータセットの実際の画像
        real_inputs = imgs.to(device)
        #本物の判定結果を出力
        real_outputs = D(real_inputs)
        real_label = torch.ones(batch_size, 1, 1, 1).to(device)

        #ランダムノイズから偽物データを生成する
        noise = noise.to(device)
        fake_inputs = G(noise)
        #偽物の判定結果を出力
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(batch_size, 1, 1, 1).to(device)

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
        noise = noise.to(device)
        fake_inputs = G(noise)
        #作った偽物を識別器に渡して判定結果を出力させる(騙せたかを確認)
        fake_outputs = D(fake_inputs)
        #作った偽物の答え(騙したい数字の情報)
        fake_targets = torch.ones(batch_size, 1, 1, 1).to(device)
        #騙せたかどうかの結果と騙したい数字を比較させてパラメータを更新
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if i % 100 == 0 or i == len(train_loader):
            print("Epoch:", epoch)
            print("discriminator_loss:", D_loss.item())
            print("generator_loss:", G_loss.item())
    
    #エポック10回毎に生成した偽物を表示
    if epoch % 10 == 0 or epoch == epoch-1:
        noise = torch.rand(100, 100, 1, 1)
        noise = noise.to(device)
        fake_inputs = G(noise)
        print("fake")
        for i in range(10):
            ax = plt.subplot(1, 10, i+1)
            ax.imshow(fake_inputs[i].detach().numpy().reshape(28,28))
        plt.show()
