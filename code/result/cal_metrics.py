import numpy as np
import torch
import os

import sys
sys.path.append('..')
from model import Encoder_Decoder_model

def cal_mse(outputs, labels):
    total_mse = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        total_mse += np.sum((outputs[i][mask] - labels[i][mask]) ** 2)
        total_count += np.sum(mask)
    return total_mse / total_count if total_count > 0 else 0

def cal_mae(outputs, labels):
    total_mae = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        total_mae += np.sum(np.abs(outputs[i][mask] - labels[i][mask]))
        total_count += np.sum(mask)
    return total_mae / total_count if total_count > 0 else 0

def cal_mre(outputs, labels):
    total_mre = 0
    total_count = 0
    for i in range(len(outputs)):
        mask = (outputs[i] > 0) & (labels[i] > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mre = (np.abs(outputs[i][mask] - labels[i][mask])) / (10 + np.fmax(outputs[i][mask], labels[i][mask])) * 100
            mre[np.isnan(mre)] = 0  # Handle division by zero or NaNs
        total_mre += np.sum(mre)
        total_count += np.sum(mask)
    return total_mre / total_count if total_count > 0 else 0

def return_mse_mae_mre(data,label):
    # outputを1次元配列にしてリストに入れる
    outputs = []
    for i in range(len(data)):
        output_i = model(data[i:i + 1]).detach().cpu().numpy().ravel()

        # ハイパス&ローパスフィルタ
        output_i[output_i<50] = 0
        output_i[output_i>8000] = 8000

        outputs.append(output_i)
        print(i)

    # labelを1次元配列にしてリストに入れる
    labels = []
    for i in range(len(label)):
        label_i = label[i:i + 1].detach().cpu().numpy().ravel()
        labels.append(label_i)
        print(i)

    # mse,mae,mreの計算
    mse = cal_mse(outputs,labels)
    mae = cal_mae(outputs,labels)
    mre = cal_mre(outputs,labels)

    return mse,mae,mre


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習済みモデルをロードする
model = Encoder_Decoder_model(inputDim=3, outputDim=1)
model = model.to(device)

# マルチGPUをONにする
'''
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print('マルチGPUの使用をONにしました')
'''
model.load_state_dict(torch.load('./model_weight.pth'))

# dataをloadする(float32→tensor→to.device)
tr_data = torch.from_numpy((np.load('./tr_data.npy')).astype(np.float32)).to(device)
tr_label = torch.from_numpy((np.load('./tr_label.npy')).astype(np.float32)).to(device)
va_data = torch.from_numpy((np.load('./va_data.npy')).astype(np.float32)).to(device)
va_label = torch.from_numpy((np.load('./va_label.npy')).astype(np.float32)).to(device)

# 学習データに対する指標
tr_metrics = return_mse_mae_mre(tr_data,tr_label)
# テストデータに対する指標
va_metrics = return_mse_mae_mre(va_data,va_label)

# textファイルに書き出し
f = open('metrics.txt','w')

f.write('training data\n')
f.write(str(tr_metrics[0])+'\n')
f.write(str(tr_metrics[1])+'\n')
f.write(str(tr_metrics[2])+'\n')

f.write('\n')

f.write('testing data\n')
f.write(str(va_metrics[0])+'\n')
f.write(str(va_metrics[1])+'\n')
f.write(str(va_metrics[2])+'\n')

f.close()