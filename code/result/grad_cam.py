import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

import sys
sys.path.append('..')
from model import Encoder_Decoder_model

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 学習済みモデルをロードする
model = Encoder_Decoder_model(inputDim=3, outputDim=1)
model.load_state_dict(torch.load('./model_weight.pth'))
model = model.to(device)
model.eval()

'''
# ResNet モデルの読み込み
model_ = models.resnet50(pretrained=True)
model_.eval()

# ターゲットレイヤーの指定
target_layers = [model_.layer4[-1]]
'''

# dataをloadする(float32→tensor→to.device)
tr_data = torch.from_numpy((np.load('./tr_data.npy')).astype(np.float32)).to(device)
tr_label = torch.from_numpy((np.load('./tr_label.npy')).astype(np.float32)).to(device)
va_data = torch.from_numpy((np.load('./va_data.npy')).astype(np.float32)).to(device)
va_label = torch.from_numpy((np.load('./va_label.npy')).astype(np.float32)).to(device)

number = 60
data = va_data[number:number+1]
label = va_label[number:number+1]

# Grad-CAMに与えるターゲット関数の定義
class MSELossTarget():
    def __init__(self, teacher):
        self.teacher = teacher

    def __call__(self,model_output):
        MSELoss = nn.MSELoss()
        return MSELoss(model_output,self.teacher)

target = [MSELossTarget(label)]

# Grad-CAMに与えるターゲットレイヤーを何通りか試して最後に平均する
cam_list = []
for i, target_layer in enumerate([[model.RN1],[model.RN2],[model.RN3],[model.RN4],[model.RN5]]):

    # Grad-CAMの初期化
    grad_cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

    # 推論とGrad-CAMの計算
    cam = grad_cam(input_tensor=data, targets=target)
    cam_list.append(cam)

# numpyに変換してINOUT画像として保存
output = model(data)

data = data.cpu().numpy()
output = output.detach().cpu().numpy()

pre_geometry = data[0,0,:,:]
geometry = data[0,1,:,:]
contact = data[0,2,:,:]
output = output[0,0,:,:]

# INOUTを画像として保存
plt.imshow(pre_geometry)
plt.axis('off')
plt.savefig("grad_cam/"+str(number)+"/INOUT/pre_geometry.jpg", bbox_inches='tight', pad_inches=0)

plt.imshow(geometry)
plt.axis('off')
plt.savefig("grad_cam/"+str(number)+"/INOUT/geometry.jpg", bbox_inches='tight', pad_inches=0)

plt.imshow(contact)
plt.axis('off')
plt.savefig("grad_cam/"+str(number)+"/INOUT/contact.jpg", bbox_inches='tight', pad_inches=0)

plt.imshow(output)
plt.axis('off')
plt.savefig("grad_cam/"+str(number)+"/INOUT/output.jpg", bbox_inches='tight', pad_inches=0)

# チャンネル次元を追加して3次元のテンソルに変換
pre_geometry = np.expand_dims(pre_geometry, axis=2)
pre_geometry = np.concatenate([pre_geometry] * 3, axis=2)

geometry = np.expand_dims(geometry, axis=2)
geometry = np.concatenate([geometry] * 3, axis=2)

contact = np.expand_dims(contact, axis=2)
contact = np.concatenate([contact] * 3, axis=2)

for i in range(5):
    cam = cam_list[i]
    cam = cam[0, :, :]
    # cam = np.expand_dims(cam, axis=2)
    # cam = np.concatenate([cam]*3, axis=2)

    pre_geometry_cam_image = show_cam_on_image(pre_geometry, cam, use_rgb=True)
    pre_geometry_cam_image = Image.fromarray(pre_geometry_cam_image)
    # pre_geometry_cam_image.show()

    geometry_cam_image = show_cam_on_image(geometry, cam, use_rgb=True)
    geometry_cam_image = Image.fromarray(geometry_cam_image)
    # geometry_cam_image.show()

    contact_cam_image = show_cam_on_image(contact, cam, use_rgb=True)
    contact_cam_image = Image.fromarray(contact_cam_image)
    # contact_cam_image.show()

    # 画像を保存する
    pre_geometry_cam_image.save("grad_cam/"+str(number)+"/RN"+str(i+1)+"/pre_geometry.jpg")
    geometry_cam_image.save("grad_cam/"+str(number)+"/RN"+str(i+1)+"/geometry.jpg")
    contact_cam_image.save("grad_cam/"+str(number)+"/RN"+str(i+1)+"/contact.jpg")

# averageの表示
ave_cam = (cam_list[0]+cam_list[1]+cam_list[2]+cam_list[3]+cam_list[4])/5
ave_cam = ave_cam[0, :, :]

pre_geometry_cam_image = show_cam_on_image(pre_geometry, ave_cam, use_rgb=True)
pre_geometry_cam_image = Image.fromarray(pre_geometry_cam_image)
# pre_geometry_cam_image.show()

geometry_cam_image = show_cam_on_image(geometry, ave_cam, use_rgb=True)
geometry_cam_image = Image.fromarray(geometry_cam_image)
# geometry_cam_image.show()

contact_cam_image = show_cam_on_image(contact, ave_cam, use_rgb=True)
contact_cam_image = Image.fromarray(contact_cam_image)
# contact_cam_image.show()

# 画像を保存する
pre_geometry_cam_image.save("grad_cam/" + str(number) + "/AVE/pre_geometry.jpg")
geometry_cam_image.save("grad_cam/" + str(number) + "/AVE/geometry.jpg")
contact_cam_image.save("grad_cam/" + str(number) + "/AVE/contact.jpg")