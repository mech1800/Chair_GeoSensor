import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('..')

from model import Encoder_Decoder_model

# ----------モデルの出力を一例表示する----------

# 0値を透明にする自作カラーマップの定義
cmap = plt.cm.jet
cmap.set_bad((0,0,0,0))  # 無効な値に対応する色

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# 学習済みモデルをロードする
model = Encoder_Decoder_model(inputDim=3, outputDim=1)
model.load_state_dict(torch.load('./model_weight.pth'))
model = model.to(device)

# dataをloadする(float32→tensor→to.device)
data = torch.from_numpy((np.load('./va_data.npy')).astype(np.float32)).to(device)
label = torch.from_numpy((np.load('./va_label.npy')).astype(np.float32)).to(device)
# シャッフルしたい場合
shuffle_index = torch.randperm(len(data))
data = data[shuffle_index]
label = label[shuffle_index]

output = model(data[0:1]).detach().cpu().numpy()
plt.imshow(output[0,0])

label = label[0:1].detach().cpu().numpy()
plt.imshow(label[0,0])

# matplotlibで扱うためにnumpyに戻す
data = data[0:1].detach().cpu().numpy()


output = np.ma.masked_where(output == 0, output)
label = np.ma.masked_where(label == 0, label)


fig = plt.figure(figsize=(5,2))
fig.subplots_adjust(hspace=0.6, wspace=0.2)

max = np.max(label)

# output用
ax1 = fig.add_subplot(1,2,1)

im1 = ax1.imshow(data[0][1], cmap='gray_r', alpha=0.2, vmin=0, vmax=1)
im1 = ax1.imshow(output[0][0], cmap=cm.jet, alpha=1, vmin=0, vmax=max)

divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
cbar1 = fig.colorbar(im1, cax1)
cbar1.ax.tick_params(labelsize=5)
cbar1.ax.set_ylim(0, max)

ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('predict', fontsize=12)

# label用
ax2 = fig.add_subplot(1,2,2)

im2 = ax2.imshow(data[0][1], cmap='gray_r', alpha=0.2, vmin=0, vmax=1)
im2 = ax2.imshow(label[0][0], cmap=cm.jet, alpha=1, vmin=0, vmax=max)

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size='5%', pad=0.05)
cbar2 = fig.colorbar(im2, cax2)
cbar2.ax.tick_params(labelsize=5)
cbar2.ax.set_ylim(0, max)

ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title('label', fontsize=12)

plt.subplots_adjust(left=0, right=0.95, top=0.85, bottom=0.05)

fig.savefig('test.png', dpi=600)

# 描画する
# plt.show()


# ----------データセットの一例を表示する----------

pre_geometry_tr = np.load('../../data/dataset/train/pre_geometry.npy')
geometry_tr = np.load('../../data/dataset/train/geometry.npy')
contact_tr = np.load('../../data/dataset/train/contact.npy')
force_tr = np.load('../../data/dataset/train/force.npy')

geo = geometry_tr[33]
pre_geo = pre_geometry_tr[33]
contact = contact_tr[33]
force = force_tr[33]

# FigureオブジェクトとAxesオブジェクトを作成
fig = plt.figure()

fig = plt.figure(figsize=(4,1))
fig.subplots_adjust(hspace=0.1, wspace=0.5)

ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)

# 画像をAxesオブジェクトに表示
ax1.imshow(geo)
ax2.imshow(pre_geo)
ax3.imshow(contact)
ax4.imshow(force)

# 軸や目盛りを非表示にする
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')

# タイトルを設定する
ax1.set_title("geometry", loc='center', fontsize=10)
ax2.set_title("pre_geometry", loc='center', fontsize=10)
ax3.set_title("contact", loc='center', fontsize=10)
ax4.set_title("force", loc='center', fontsize=10)

plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)

# グラフを表示
plt.show()
