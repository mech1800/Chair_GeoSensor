import numpy as np
from dataset import MyDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Encoder_Decoder_model
import torch.optim as optim
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# パラメータ
batchsize = 16
lr = 0.001
weight_decay=0.01
epochs = 100


# データセットの読み込み
pre_geometry_tr = np.load('../data/dataset/train/pre_geometry.npy')
geometry_tr = np.load('../data/dataset/train/geometry.npy')
contact_tr = np.load('../data/dataset/train/contact.npy')
force_tr = np.load('../data/dataset/train/force.npy')

pre_geometry_va = np.load('../data/dataset/test/pre_geometry.npy')
geometry_va = np.load('../data/dataset/test/geometry.npy')
contact_va = np.load('../data/dataset/test/contact.npy')
force_va = np.load('../data/dataset/test/force.npy')

# 学習データとテストデータを作成する
tr_data = np.stack([pre_geometry_tr, geometry_tr, contact_tr], axis=1)
tr_label = np.reshape(force_tr, [force_tr.shape[0], -1, force_tr.shape[1], force_tr.shape[2]])

va_data = np.stack([pre_geometry_va, geometry_va, contact_va], axis=1)
va_label = np.reshape(force_va, [force_va.shape[0], -1, force_va.shape[1], force_va.shape[2]])

# resultに保存する
np.save('result/tr_data', tr_data)
np.save('result/tr_label', tr_label)
np.save('result/va_data', va_data)
np.save('result/va_label', va_label)

# 学習データをイテレータにする
dataset = MyDataset(tr_data, tr_label)
trainloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

# テストデータをイテレータにする
dataset = MyDataset(va_data, va_label)
validloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)


# 訓練用関数
def train(model, device, criterion, optimizer, trainloader):
    model.train()
    running_loss = 0

    for data, label in trainloader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            running_loss += loss.item()

    return running_loss/len(trainloader)


# 評価用関数
def valid(model, device, criterion, validloader):
    model.eval()
    running_loss = 0

    for data, label in validloader:
        data = data.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label)
            running_loss += loss.item()

    return running_loss/len(validloader)


# モデル，評価関数，最適化関数を呼び出す
model = Encoder_Decoder_model(inputDim=3, outputDim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 指定したエポック数だけ学習指せる
tr_loss = []
va_loss = []

for epoch in range(1, 1+epochs):
    loss = train(model, device, criterion, optimizer, trainloader)
    tr_loss.append(loss)

    loss = valid(model, device, criterion, validloader)
    va_loss.append(loss)

    print(str(epoch)+'epoch通過')

else:
    torch.save(model.state_dict(), 'result/model_weight.pth')


# lossの推移をグラフにする
x = [i for i in range(epochs)]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, tr_loss, label='tr_loss')
ax.plot(x, va_loss, label='va_loss')
ax.set_xlabel('epoch')
ax.set_ylabel('MSE_loss')
ax.legend(loc='upper right')
fig.savefig('result/loss.png')
plt.show()