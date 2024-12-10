from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim

# データセットの準備
fash_train = FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
fash_test = FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())

fashion_class = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

train_loader = DataLoader(fash_train, batch_size=60, shuffle=True)
test_loader = DataLoader(fash_test, batch_size=60, shuffle=False)

# モデル定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  #自動で形状を推定
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデルと損失関数
net = Net()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# 学習ループ
record_loss_train = []
record_loss_test = []

for i in range(20):
    net.train()
    loss_train = 0
    for j, (x, t) in enumerate(train_loader):
        y = net(x)
        loss = loss_func(y, t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train /= j + 1
    record_loss_train.append(loss_train)

    net.eval()
    loss_test = 0
    for j, (x, t) in enumerate(test_loader):
        y = net(x)
        loss = loss_func(y, t)
        loss_test += loss.item()

    loss_test /= j + 1
    record_loss_test.append(loss_test)
    print(f"Epoch: {i}, Loss_train: {loss_train:.4f}, Loss_test: {loss_test:.4f}")

plt.plot(range(len(record_loss_train)), record_loss_train, label="loss_train")
plt.plot(range(len(record_loss_test)), record_loss_test, label="loss_test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


fashionM_loader = DataLoader(fash_test, batch_size=1, shuffle=True)
dataiter = iter(fashionM_loader)
images, labels = next(dataiter)

plt.imshow(images[0].squeeze(), cmap="gray")
plt.show()

net.eval()
y = net(images)
print(f"正解: {fashion_class[labels[0].item()]}, 推論: {fashion_class[y.argmax().item()]}")