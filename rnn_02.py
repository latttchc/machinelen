from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset
import torch


#データのダウンロード
Fashion_train = FashionMNIST("./data",
                       train = True,
                       download = True,
                       transform = transforms.ToTensor())

print("データの数:",len(Fashion_train))


img_size = 28 #画像サイズ　28×28ピクセル
n_time = 14 #画像の上半分の入力から下半分を生成させることを考える
n_mid = 256 #RNNの中間層のニューロン数(適当)
n_in = img_size #入力画像の次元
n_out = img_size #出力画像の次元
n_sample_in_img = img_size - n_time #サンプル数

#データの準備
dataloader = DataLoader(Fashion_train,
                        batch_size = len(Fashion_train),
                        shuffle = True)
#イテレータで1枚ずつ取り出して形を変換する→train_imgsへ格納
dataiter = iter(dataloader)
train_imgs, labels = dataiter.__next__()
train_imgs = train_imgs.view(-1,img_size, img_size)
n_sample = len(train_imgs)*n_sample_in_img

#学習データと正解データの変数の定義
input_data = torch.zeros((n_sample, n_time, n_in))
correct_data = torch.zeros((n_sample,n_out))

#各行を時刻とみなし、各行の列方向の画素の並びから
#学習データ(複数の画素シーケンス)と正解データ（ひとつ隣の画素）を作成する
for i in range(len(train_imgs)):
    for j in range(n_sample_in_img):
        sample_id = i*n_sample_in_img+j
        input_data[sample_id] = train_imgs[i, j:j+n_time]
        correct_data[sample_id] = train_imgs[i, j+n_time]
dataset = TensorDataset(input_data, correct_data)

#作成した学習データと正解データからデータローダでバッチ学習できるようにする
train_loader = DataLoader(dataset,
                          batch_size = 128,
                          shuffle = True)

#テスト用データの準備
n_disp = 10 #10枚の画像に対して下半分を生成させる
Fasion_test = FashionMNIST("./data",
                       train = False,
                       download = True,
                       transform = transforms.ToTensor())
test_loader = DataLoader(Fasion_test,
                         batch_size = n_disp,
                         shuffle = False)
dataiter = iter(test_loader)
test_imgs, labels = dataiter.__next__()
test_imgs = test_imgs.view(-1,img_size, img_size)

#モデルの構築
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.RNN(input_size = n_in,#入力次元は画像サイズ
                            hidden_size = n_mid,#中間層のニューロン数
                            batch_first = True) #バッチを先頭に
        self.fc = nn.Linear(n_mid, n_out) #全結合層の定義

    def forward(self,x):
        y_rnn, h = self.lstm(x,None)
        y = self.fc(y_rnn[:, -1, :]) #-1で最後のデータを出力させる
        return y
    
#ネットワークをインスタンス
net = Net()
print(net)
#もしGPUを搭載していてCUDAが使えるなら#を外して下記を実行する
#net.cuda()

#学習結果から画像を生成する関数
def img_gen():
    #テストに使う元画像10枚を表示
    print("original:")
    plt.figure(figsize=(20,2))
    for i in range(n_disp):
        ax = plt.subplot(1, n_disp, i+1)
        ax.imshow(test_imgs[i])
    plt.show()

    #学習結果から下半分を生成
    print("generated:")
    net.eval() #評価モード
    gen_imgs = test_imgs.clone() #元画像と同じサイズでメモリ確保
    plt.figure(figsize=(20,2))
    for i in range(n_disp):
        for j in range(n_sample_in_img):
            #元画像の上半分のみをコピーして格納
            x = gen_imgs[i,j:j+n_time].view(1,n_time,img_size)
            #x = x.cuda() #もしGPUを搭載していてCUDAが使えれば有効化する
            #元画像の上半分を入力して下半分を生成
            gen_imgs[i,j+n_time] = net(x)[0]
        #生成結果を表示
        ax = plt.subplot(1,n_disp,i+1)
        ax.imshow(gen_imgs[i].detach())
    plt.show()


#損失関数を定義する
loss_func = nn.MSELoss()

#最適化手法としてAdamを用いる
optimizer = optim.Adam(net.parameters())

#損失関数の値のログ
record_loss_train = []

#学習 エポック数は25とする
epochs = 25
for i in range(epochs):
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
    #損失関数の値を表示
    print("Epoch:", i, "Loss_train:", loss_train)
    if i%5==0 or i==epochs-1:
        #エポック5回毎に画像生成
        img_gen()
        
#グラフの作成
plt.plot(range(len(record_loss_train)),#横軸の設定
         record_loss_train, #縦軸の値の設定
         label = "loss_train") #グラフのラベル付け

plt.xlabel("Epoch")
plt.ylabel("Loss_func_value")
plt.show()
