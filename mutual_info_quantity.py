# calcMI_funcに統合済み

import os
from func_3Dver import crop_img, img_MI
from PIL import Image
import numpy as np
from global_value import EPOCHS, BATCH_SIZE, dataset_num, date, train_data_date

x = 120
y = 120
bytesize = 1
histgramX = 256
histgramY = 256
global crop_savefile
global pre_filepath
"""
# file_info
# このmodelで復元された画像の相互情報量を求める
####################
EPOCHS = 10001
BATCH_SIZE = 16
dataset_num = 58 * 100
date = "20230120"
pre_file = 'adler_data'

#####################

train_data_date = "20230120"
"""


def mutual_info_quantity(pre_file, t_v):
    global crop_savefile
    global pre_filepath

    data1, data2 = data_read(pre_file, t_v)

    # 配列の領域確保
    histmap = np.zeros((histgramX, histgramY))
    p_ab = np.zeros((histgramX, histgramY))
    p_a = np.zeros(histgramX)
    p_b = np.zeros(histgramY)

    histsum = 0

    # 2次元ヒストグラム作成
    for k in range(len(data1)):
        num1 = data1[k]
        num2 = data2[k]

        histmap[num1][num2] += 1

    for m in range(histgramX):
        for n in range(histgramY):
            histsum += histmap[m][n]

    # 相互情報量計算
    # p(a,b)
    for m in range(histgramX):
        for n in range(histgramY):
            p_ab[m][n] = histmap[m][n] / (1.0 * histsum)

    # p(a), p(b)
    for m in range(histgramX):
        for n in range(histgramY):
            p_a[m] += p_ab[m][n]
            p_b[n] += p_ab[m][n]

    """
    for m in range(histgramX):
        for n in range(histgramY):
            if p_a[m]*p_b[n] != 0:
                t[m][n] = p_ab[m][n]/(1.0*p_a[m]*p_b[n])
                if t[m][n] != 0:
                    mutual += p_a[m][n]*math.log(t[m][n], 2)
    """

    # e(a):aのエントロピー, e(b):bのエントロピー, e(a, b):a,bの結合エントロピー
    e_a = 0
    e_b = 0
    e_ab = 0
    for m in range(histgramX):
        if p_a[m] != 0:
            e_a -= p_a[m] * np.log2(p_a[m])
    for n in range(histgramY):
        if p_b[n] != 0:
            e_b -= p_b[n] * np.log2(p_b[n])
    for m in range(histgramX):
        for n in range(histgramY):
            if p_ab[m][n] != 0:
                e_ab -= p_ab[m][n] * np.log2(p_ab[m][n])

    mutual = e_a + e_b - e_ab
    print(f"e_a:{e_a}, e_b:{e_b}, e_ab:{e_ab}")
    print(mutual)

    # 図の作成
    # send_data
    # 左：original 右：predict
    send_img_data = [round(e_b, 4), round(mutual, 4)]
    send_dir_name = [crop_savefile, pre_filepath]

    # save_folderの作成
    save_dir_name = date + r"\\predict\mutual_info\\"  # prdict_k20
    save_dir_name = save_dir_name + f"dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}"
    os.makedirs(save_dir_name, exist_ok=True)
    save_dir_name = save_dir_name + f"\\{pre_file}"

    # 教師画像と復元画像並べた図の保存
    # img_MI(send_img_data, send_dir_name, save_dir_name)

    return e_b, mutual  # e_b:original画像の平均情報量, mutual:相互情報量


def data_read(pre_file, t_v):
    global pre_filepath
    global crop_savefile

    dirname1 = date + "\\predict\\" + t_v + "_predict\\"  # predict_k20 , train
    pre_filepath = dirname1 + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_one.jpg'
    dirname2 = "train_data\\" + train_data_date + "\\" + pre_file + "\\image\\img0\\img0_0.jpg"

    # フォルダ作成
    mi_dirname = dirname1 + "mutual_info\\"
    os.makedirs(mi_dirname, exist_ok=True)

    # train_data サイズ変更 128*128 -> 120*120
    crop_savefile = mi_dirname + pre_file + "_resize.jpg"
    crop_img(crop_savefile, dirname2, 120)

    data1 = np.asarray(Image.open(pre_filepath))
    data1 = np.ravel(data1).astype(int)

    data2 = np.asarray(Image.open(crop_savefile))
    data2 = np.ravel(data2).astype(int)
    return data1, data2  # data1:予測画像　data2:一層目の画像


