import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

# GPU確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_image_dir = './data/Flower'

# 画像変換
image_size = 64
data_transform = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
}

train_set = datasets.ImageFolder(root=train_image_dir, transform=data_transform['train'])
batch_size = 28
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 識別器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 512)
        self.fc2 = nn.Linear(512, 1)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 64 * 64 * 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 3, 64, 64)
        return x

G = Generator().to(device)
D = Discriminator().to(device)

loss_fn = nn.BCELoss()

lr = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

#損失関数の値のログ
record_loss_D = []
record_loss_G = []

# 学習
epochs = 50
for epoch in range(epochs):
    loss_D = 0
    loss_G = 0
    for i, (imgs, _) in enumerate(train_loader):
        real_inputs = imgs.to(device)
        real_label = torch.ones(real_inputs.size(0), 1).to(device)
        fake_label = torch.zeros(real_inputs.size(0), 1).to(device)
        
        # 識別器の学習
        real_outputs = D(real_inputs)
        real_loss = loss_fn(real_outputs, real_label)
        
        noise = torch.randn(real_inputs.size(0), 128).to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs.detach())
        fake_loss = loss_fn(fake_outputs, fake_label)
        
        D_loss = real_loss + fake_loss
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        
        # 生成器の学習
        fake_outputs = D(fake_inputs)
        G_loss = loss_fn(fake_outputs, real_label)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
        loss_D += D_loss.item()
        loss_G += G_loss.item()



        if i % 100 == 0 or i == len(train_loader) - 1:
            print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")
    
    loss_D /= epoch+1
    loss_G /= epoch+1
    
    record_loss_D.append(loss_D)
    record_loss_G.append(loss_G)

    

    # 10エポックごとに生成画像を表示
    if epoch % 10 == 0 or epoch == epochs - 1:
        noise = torch.randn(10, 128).to(device)
        fake_images = G(noise).detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 10, figsize=(15, 5))
        for j in range(10):
            axes[j].imshow((fake_images[j].transpose(1, 2, 0) + 1) / 2)
            axes[j].axis('off')
        plt.show()
    
plt.plot(range(len(record_loss_D)),
         record_loss_D,
         label = "D Loss")

plt.plot(range(len(record_loss_G)),
         record_loss_G,
         label = "G Loss")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss_func_value")
plt.show()
    




