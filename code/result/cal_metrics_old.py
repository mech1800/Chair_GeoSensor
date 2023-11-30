import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('..')

from model import Encoder_Decoder_model

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# 学習済みモデルをロードする
model = Encoder_Decoder_model(inputDim=3, outputDim=1)
model.load_state_dict(torch.load('./model_weight.pth'))
model = model.to(device)

# dataをloadする(float32→tensor→to.device)
tr_data = torch.from_numpy((np.load('./tr_data.npy')).astype(np.float32)).to(device)
tr_label = torch.from_numpy((np.load('./tr_label.npy')).astype(np.float32)).to(device)
va_data = torch.from_numpy((np.load('./va_data.npy')).astype(np.float32)).to(device)
va_label = torch.from_numpy((np.load('./va_label.npy')).astype(np.float32)).to(device)

'''
# テスト
output = model(tr_data[11:12]).detach().cpu().numpy()
plt.imshow(output[0,0])
plt.show()

label = tr_label[11:12].detach().cpu().numpy()
plt.imshow(label[0,0])
plt.show()

output = model(va_data[11:12]).detach().cpu().numpy()
plt.imshow(output[0,0])
plt.show()

label = va_label[11:12].detach().cpu().numpy()
plt.imshow(label[0,0])
plt.show()

a = 1
'''

def cal_mse(outputs,labels):
    total_mse = 0
    for i in range(len(outputs)):
        total_mse += np.mean((outputs[i]-labels[i])**2)
    return total_mse/len(outputs)

def cal_mae(outputs,labels):
    total_mae = 0
    for i in range(len(outputs)):
        total_mae += np.mean(np.abs(outputs[i]-labels[i]))
    return total_mae/len(outputs)

def cal_mre(outputs,labels):
    total_mre = 0
    for i in range(len(outputs)):
        total_mre += np.mean((np.abs(outputs[i]-labels[i]))/(0.01+np.fmax(outputs[i],labels[i]))*100)
    return total_mre/len(outputs)

'''
def cal_tmre(outputs,labels,count):
    total_tmre = 0
    for i in range(len(outputs)):
        error = np.sum((np.abs(outputs[i]-labels[i]))/(1+np.fmax(outputs[i],labels[i]))*100)/(1+np.sum(count[i]))
        if np.isnan(error).any():
            pass
        else:
            total_tmre += error
    return total_tmre/len(outputs)
'''

def cal_tmre(outputs,labels):
    total_tmre = 0

    for i in range(len(outputs)):
        indices = np.where(outputs[i] > 0)[0]
        count = len(indices)

        output = outputs[i][indices]
        label = labels[i][indices]

        error = np.sum((np.abs(output-label))/(10+np.fmax(output,label))*100)/(10+np.sum(count))
        if np.isnan(error).any():
            pass
        else:
            total_tmre += error
    return total_tmre/len(outputs)

def cal_masked_tmre(outputs,labels):
    total_tmre = 0

    for i in range(len(outputs)):
        indices = np.where(labels[i] > 0)[0]
        count = len(indices)

        output = outputs[i][indices]
        label = labels[i][indices]

        error = np.sum((np.abs(output-label))/(10+np.fmax(output,label))*100)/(10+np.sum(count))
        if np.isnan(error).any():
            pass
        else:
            total_tmre += error
    return total_tmre/len(outputs)

def return_mse_mae_mre_tmre(data,label):
    # outputを1次元配列にしてリストに入れる
    outputs = []
    for i in range(len(data)):
        output_i = model(data[i:i+1]).detach().cpu().numpy().ravel()
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

    # tmreの計算
    tmre = cal_tmre(outputs,labels)
    masked_tmre = cal_masked_tmre(outputs, labels)

    return mse,mae,mre,tmre,masked_tmre

# 学習データに対する指標
tr_metrics = return_mse_mae_mre_tmre(tr_data,tr_label)

# テストデータに対する指標
va_metrics = return_mse_mae_mre_tmre(va_data,va_label)

# textファイルに書き出し
f = open('metrics.txt','w')

f.write('training data\n')
f.write(str(tr_metrics[0])+'\n')
f.write(str(tr_metrics[1])+'\n')
f.write(str(tr_metrics[2])+'\n')
f.write(str(tr_metrics[3])+'\n')
f.write(str(tr_metrics[4])+'\n')

f.write('\n')

f.write('testing data\n')
f.write(str(va_metrics[0])+'\n')
f.write(str(va_metrics[1])+'\n')
f.write(str(va_metrics[2])+'\n')
f.write(str(va_metrics[3])+'\n')
f.write(str(va_metrics[4])+'\n')

f.close()