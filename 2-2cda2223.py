import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# GPU確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#データの設定 MNIST
batch_size = 64
train_set = datasets.MNIST('data/',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True)

save_folder = "fake_images"

#識別器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
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
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)

G = generator().to(device)
D = discriminator().to(device)


loss = nn.BCELoss()

lr = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))



#学習　
def learn():
    epochs = 50
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            i += 1

            #識別
            real_inputs = imgs.to(device)
            real_outputs = D(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            noise = (torch.rand(real_inputs.shape[0],128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = G(noise)

            fake_outputs = D(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)

            D_loss = loss(outputs, targets)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()


            #生成
            noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
            noise = noise.to(device)
            fake_inputs = G(noise)
            fake_outputs = D(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = loss(fake_outputs, fake_targets)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            if i % 100 == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")
                
        if epoch % 50 == 0:
            noise = torch.randn(10, 128).to(device) 
            fake_inputs = G(noise).detach().cpu() 
            fake_inputs = (fake_inputs + 1) / 2 

            batches_done = epoch * len(train_loader) + i
            os.makedirs(save_folder, exist_ok=True) 

            for idx in range(10):
                save_path = f"{save_folder}/{batches_done}_{idx}.png"
                save_image(fake_inputs[idx], save_path)

# ユーザ
def game():
    score = 0
    rounds = 5
    
    fake_images_list = [f for f in os.listdir(save_folder) if f.endswith(".png")]

    for _ in range(rounds):
        is_real = random.choice([True, False]) 
        if is_real:
            image, _ = random.choice(train_set)  
            image = image.squeeze().numpy() 
        else:
            fake_image_path = os.path.join(save_folder, random.choice(fake_images_list))
            image = Image.open(fake_image_path)
            image = np.array(image) / 255.0  

        plt.imshow(image, cmap='gray')
        plt.axis("off")
        plt.show()
        
        user_input = input("この画像は本物？ (y:本物 / n:偽物): ").lower()
        user_guess = (user_input == "y")  
        
        if user_guess == is_real:
            print("正解 🎉")
            score += 1
        else:
            print("不正解 😢")
        
        print(f"実際の画像: {'本物' if is_real else '偽物'}")
        print("-" * 30)
    
    print(f"ゲーム終了！ あなたのスコア: {score}/{rounds}")


def main():
    mode = input("モードを選択!(学習させないと判別できません) 1: 学習  2: ゲームモード ==> ")
    if mode == '1':
        learn()
    elif mode == '2':
        game()
    else:
        print("1か2を選択してください.")

if __name__ == "__main__":
    main()

    



