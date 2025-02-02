import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image


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


def learn():
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

            if i % 100 == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")
        

        # 10エポックごとに生成画像を表示
        if epoch % 10 == 0 or epoch == epochs - 1:
            noise = torch.randn(28, 128).to(device)
            fake_images = G(noise).detach().cpu().numpy()
            
            fig, axes = plt.subplots(1, 10, figsize=(15, 5))
            for j in range(10):
                axes[j].imshow((fake_images[j].transpose(1, 2, 0) + 1) / 2)
                axes[j].axis('off')
            plt.show()

def save_fake_images(fake_folder, G, device):
    if not os.path.exists(fake_folder):
        os.makedirs(fake_folder)
    
    noise = torch.randn(10, 128).to(device)
    fake_images = G(noise).detach().cpu().numpy()
    
    for i, img in enumerate(fake_images):
        img = (img.squeeze() + 1) / 2 * 255  # 画像を0-255の範囲に正規化
        img = Image.fromarray(img[0].astype('uint8'))
        img.save(os.path.join(fake_folder, f"fake_{i}.png"))
    
    print(f"偽物の画像を {fake_folder} に保存しました。")

def load_fake_images(fake_folder):
    if not os.path.exists(fake_folder):
        os.makedirs(fake_folder)
        print(f"フォルダ '{fake_folder}' を作成しました。偽物の画像を入れてください。")
        return []

    fake_images = []
    for filename in os.listdir(fake_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(os.path.join(fake_folder, filename)).convert('L')
            img = transforms.ToTensor()(img).numpy()[0]
            fake_images.append(img)
    return fake_images

def user_judgment(fake_folder):
    fake_images = load_fake_images(fake_folder)
    noise = torch.randn(1, 128).to(device)
    fake_image = fake_images[random.randint(0, len(fake_images) - 1)] if fake_images else G(noise).detach().cpu().numpy()[0]
    real_image, _ = random.choice(train_set)
    real_image = real_image.numpy()[0]
    
    is_real = random.choice([True, False])
    image = real_image if is_real else fake_image
    
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    
    user_input = input("これは本物？(y/n): ")
    user_guess = user_input.lower() == 'y'
    
    input_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
    D_result = D(input_tensor).item()
    model_decision = D_result > 0.5
    
    print(f"ユーザの答え: {'本物' if user_guess else '偽物'}")
    print(f"実際の答え: {'本物' if is_real else '偽物'}")
    print(f"識別器の判定: {'本物' if model_decision else '偽物'} (信頼度: {D_result:.2f})")
    
    if user_guess == is_real:
        print("正解！")
    else:
        print("不正解！")

def main():
    mode = input("モードを選択してください (1: 学習, 2: 判別): ")
    fake_folder = "fake_images"
    
    if mode == '1':
        learn()
        save_fake_images(fake_folder, G, device)
    elif mode == '2':
        user_judgment(fake_folder)
    else:
        print("無効な入力です。1 または 2 を選択してください。")

if __name__ == "__main__":
    main()