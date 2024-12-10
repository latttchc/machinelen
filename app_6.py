import torch.nn as nn 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #畳み込み層の定義
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)

        #プーリング層の定義
        self.pool = nn.MaxPool2d(2, 2)

        #ドロップアウトの定義
        self.dropout = nn.Dropout(p=0.5)

        #活性化関数の定義
        self.relu = nn.ReLU()

        #全結合層の定義
        self.fc1 = nn.Linear(16*5*5, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
net = Net()
print(net)
