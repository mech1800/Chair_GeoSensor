import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math


'''
geo = np.load('dataset/train/geometry.npy')
pre_geo = np.load('dataset/train/pre_geometry.npy')
contact = np.load('dataset/train/contact.npy')
force = np.load('dataset/train/force.npy')

for i in [0,28,37,55]:
    plt.imshow(geo[i])
    plt.show()

    plt.imshow(pre_geo[i])
    plt.show()

    plt.imshow(contact[i])
    plt.show()

    plt.imshow(force[i])
    plt.show()
'''


def determine_offset(scale):
    LR = np.random.uniform(-50*(scale-1), 50*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if LR>0:
        LR = math.floor(LR)
    else:
        LR = math.ceil(LR)

    UD = np.random.uniform(-50*(scale-1), 50*(scale-1))
    # 絶対値が小さくなるように切り捨て
    if UD>0:
        UD = math.floor(UD)
    else:
        UD = math.ceil(UD)

    return (LR,UD)

def randam_expand_and_slide(geo,pre_geo,contact,force):
    # 画像をランダムな倍数で拡大する
    random_scale = np.random.uniform(1, 1.2)
    modified_geo = ndimage.zoom(geo, random_scale, order=0)
    modified_pre_geo = ndimage.zoom(pre_geo, random_scale, order=0)
    modified_contact = ndimage.zoom(contact, random_scale, order=0)
    modified_force = ndimage.zoom(force, random_scale, order=0)

    # 画像をランダムなオフセットでスライドさせる
    offset = determine_offset(random_scale)

    # 拡大した画像から中心からランダムなオフセットの分だけずらして100×100ピクセルを抽出する
    start_x = (modified_geo.shape[1]-100)//2
    start_y = (modified_geo.shape[0]-100)//2

    modified_geo = modified_geo[(start_y+offset[1]):(start_y+offset[1])+100, (start_x+offset[0]):(start_x+offset[0])+100]
    modified_pre_geo = modified_pre_geo[(start_y+offset[1]):(start_y+offset[1])+100, (start_x+offset[0]):(start_x+offset[0])+100]
    modified_contact = modified_contact[(start_y+offset[1]):(start_y+offset[1])+100, (start_x+offset[0]):(start_x+offset[0])+100]
    modified_force = modified_force[(start_y+offset[1]):(start_y+offset[1])+100, (start_x+offset[0]):(start_x+offset[0])+100]

    return modified_geo,modified_pre_geo,modified_contact,modified_force


# オリジナルのデータセットの読み込み
geo = np.load('original_dataset/geometry.npy')
pre_geo = np.load('original_dataset/pre_geometry.npy')
contact = np.load('original_dataset/contact.npy')
force = np.load('original_dataset/force.npy')

# オリジナルのデータセットを0次元でランダムにシャッフルする
shuffled_indices = np.random.permutation(geo.shape[0])
train_indices = shuffled_indices[0:int(len(shuffled_indices)*0.9)]
test_indices = shuffled_indices[int(len(shuffled_indices)*0.9):]


# -----学習データの作成-----
# 拡張前の学習データ
geo_train = geo[train_indices]
pre_geo_train = pre_geo[train_indices]
contact_train = contact[train_indices]
force_train = force[train_indices]

# 拡張後の学習データの保存先
geo_train_output = np.empty((0,100,100))
pre_geo_train_output = np.empty((0,100,100))
contact_train_output = np.empty((0,100,100))
force_train_output = np.empty((0,100,100))

# 各画像に対して10回ずつデータ拡張を行う
for i in range(geo_train.shape[0]):
    print(i)
    # オリジナルは最初に保存する
    geo_train_output = np.concatenate((geo_train_output,geo_train[i][np.newaxis,:]), axis=0).astype(np.float32)
    pre_geo_train_output = np.concatenate((pre_geo_train_output,pre_geo_train[i][np.newaxis,:]), axis=0).astype(np.float32)
    contact_train_output = np.concatenate((contact_train_output,contact_train[i][np.newaxis,:]), axis=0).astype(np.float32)
    force_train_output = np.concatenate((force_train_output,force_train[i][np.newaxis,:]), axis=0).astype(np.float32)

    for _ in range(10):
        geo_tmp,pre_geo_tmp,contact_tmp,force_tmp = randam_expand_and_slide(geo_train[i],pre_geo_train[i],contact_train[i],force_train[i])

        geo_train_output = np.concatenate((geo_train_output, geo_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        pre_geo_train_output = np.concatenate((pre_geo_train_output, pre_geo_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        contact_train_output = np.concatenate((contact_train_output, contact_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        force_train_output = np.concatenate((force_train_output, force_tmp[np.newaxis, :]), axis=0).astype(np.float32)

# 出力する
np.save('dataset/train/geometry.npy',geo_train_output)
np.save('dataset/train/pre_geometry.npy',pre_geo_train_output)
np.save('dataset/train/contact.npy',contact_train_output)
np.save('dataset/train/force.npy',force_train_output)

# メモリから消去
del geo_train_output
del pre_geo_train_output
del contact_train_output
del force_train_output


# -----テストデータの作成-----
# 拡張前のテストデータ
geo_test = geo[test_indices]
pre_geo_test = pre_geo[test_indices]
contact_test = contact[test_indices]
force_test = force[test_indices]

# 拡張後のテストデータの保存先
geo_test_output = np.empty((0,100,100))
pre_geo_test_output = np.empty((0,100,100))
contact_test_output = np.empty((0,100,100))
force_test_output = np.empty((0,100,100))

# 各画像に対して10回ずつデータ拡張を行う
for i in range(geo_test.shape[0]):
    print(i)
    # オリジナルは最初に保存する
    geo_test_output = np.concatenate((geo_test_output,geo_test[i][np.newaxis,:]), axis=0).astype(np.float32)
    pre_geo_test_output = np.concatenate((pre_geo_test_output,pre_geo_test[i][np.newaxis,:]), axis=0).astype(np.float32)
    contact_test_output = np.concatenate((contact_test_output,contact_test[i][np.newaxis,:]), axis=0).astype(np.float32)
    force_test_output = np.concatenate((force_test_output,force_test[i][np.newaxis,:]), axis=0).astype(np.float32)

    for _ in range(10):
        geo_tmp,pre_geo_tmp,contact_tmp,force_tmp = randam_expand_and_slide(geo_test[i],pre_geo_test[i],contact_test[i],force_test[i])

        geo_test_output = np.concatenate((geo_test_output, geo_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        pre_geo_test_output = np.concatenate((pre_geo_test_output, pre_geo_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        contact_test_output = np.concatenate((contact_test_output, contact_tmp[np.newaxis, :]), axis=0).astype(np.float32)
        force_test_output = np.concatenate((force_test_output, force_tmp[np.newaxis, :]), axis=0).astype(np.float32)

# 出力する
np.save('dataset/test/geometry.npy',geo_test_output)
np.save('dataset/test/pre_geometry.npy',pre_geo_test_output)
np.save('dataset/test/contact.npy',contact_test_output)
np.save('dataset/test/force.npy',force_test_output)

# メモリから消去
del geo_test_output
del pre_geo_test_output
del contact_test_output
del force_test_output