# mutual_info_quantityを参照し相互情報量を求める
# 複数のファイルを参照

import numpy as np
from PIL import Image
import os
import pickle
import csv
from natsort import natsorted
# ファイルのimport
from global_value import BATCH_SIZE, dataset_num, date, train_data_date
import cv2

# 　変数の定義
x = 120
y = 120
bytesize = 1
histgramX = 256
histgramY = 256
stim_head = 201 + 1
global crop_savefile
global pre_filepath

dirmain = r"F:\train_data\20240108\0to1199"


def calc_test_MI(model_name, EPOCHS):
    # unused_fileの読み込み
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\" + model_name + "\\"  # k_20
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = pickle.load(f)
    f.close()
    unused_filename = natsorted(unused_filename)

    filename_all = ["ファイル名"]
    e_b_all = ["平均情報量"]  # originalの平均情報量のリスト
    MI_all = ["相互情報量"]  # 相互情報量のリスト

    for filename in unused_filename:
        # predict_image(filename)
        e_b, mutual = mutual_info_quantity(filename, "test", model_name, EPOCHS)

        filename_all.append(filename)
        e_b_all.append(e_b)
        MI_all.append(mutual)

        print(f"{filename} is finished!!")

    # 情報量の保存
    MI_save_path = result_dirpath + r"predict\mutual_info"  # k_20
    MI_save_path = MI_save_path + f"\\mutual_info={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_test.csv"  # train

    with open(MI_save_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(filename_all)
        writer.writerow(e_b_all)
        writer.writerow(MI_all)


def calc_train_MI(model_name, EPOCHS):
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\" + model_name + "\\"
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = pickle.load(f)
    f.close()

    used_filename = natsorted(used_filename)
    filename_all = ["ファイル名"]
    e_b_all = ["平均情報量"]  # originalの平均情報量のリスト
    MI_all = ["相互情報量"]  # 相互情報量のリスト

    for filename in used_filename:
        # predict_image(filename)
        e_b, mutual = mutual_info_quantity(filename, "train", model_name, EPOCHS)

        filename_all.append(filename)
        e_b_all.append(e_b)
        MI_all.append(mutual)

        print(f"{filename} is finished!!")

    # 情報量の保存
    MI_save_path = result_dirpath + r"predict\mutual_info"  # k_20
    MI_save_path = MI_save_path + f"\\mutual_info={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_train.csv"

    with open(MI_save_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(filename_all)
        writer.writerow(e_b_all)
        writer.writerow(MI_all)


def mutual_info_quantity(pre_file, t_v, model_name, EPOCHS):
    global crop_savefile
    global pre_filepath

    data1, data2 = data_read(pre_file, t_v, model_name, EPOCHS)

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
    # print(f"e_a:{e_a}, e_b:{e_b}, e_ab:{e_ab}")
    # print(mutual)

    # 図の作成
    # send_data
    # 左：original 右：predict
    send_img_data = [round(e_b, 4), round(mutual, 4)]
    send_dir_name = [crop_savefile, pre_filepath]

    # save_folderの作成
    save_dir_name = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\" + model_name + r"\\predict\mutual_info\\"  # prdict_k20
    save_dir_name = save_dir_name + f"dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}"
    os.makedirs(save_dir_name, exist_ok=True)
    save_dir_name = save_dir_name + f"\\{pre_file}"

    # 教師画像と復元画像並べた図の保存
    # img_MI(send_img_data, send_dir_name, save_dir_name)

    return e_b, mutual  # e_b:original画像の平均情報量, mutual:相互情報量


def data_read(pre_file, t_v, model_name, EPOCHS):
    global pre_filepath
    global crop_savefile

    dirname1 = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\" + model_name + "\\predict\\" + t_v + "_predict\\"  # predict_k20 , train
    pre_filepath = dirname1 + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.jpg'
    # dirname2 = r"C:\Users\AIlab\labo\3DCNN\train_data\\" + train_data_date + "\\" + pre_file + "\\image\\img0\\img0_0.jpg"
    dirname2 = dirmain + "\\" + pre_file + f"\\image\\img0\\img0_{stim_head}.jpg"
    dirname2_npy = dirmain + "\\" + pre_file + f"\\img0\\img0_{stim_head}.npy"
    # フォルダ作成
    mi_dirname = dirname1 + "resize_img\\"
    os.makedirs(mi_dirname, exist_ok=True)

    # train_data サイズ変更 128*128 -> 120*120
    crop_savefile = mi_dirname + pre_file + "_resize.jpg"
    crop_img(crop_savefile, dirname2, 120)

    # resize_save
    os.makedirs(mi_dirname + "save_npy", exist_ok=True)
    resize_npy_save(mi_dirname + "save_npy\\" + pre_file + "_resize.npy", dirname2_npy)

    data1 = np.asarray(Image.open(pre_filepath))
    data1 = np.ravel(data1).astype(int)

    data2 = np.asarray(Image.open(crop_savefile))
    data2 = np.ravel(data2).astype(int)
    return data1, data2  # data1:予測画像　data2:一層目の画像


def crop_img(savefile, path, crop_w_h):
    im = Image.open(path)
    img_width, img_height = im.size
    pil_img = im.crop(((img_width - crop_w_h) // 2,
                       (img_height - crop_w_h) // 2,
                       (img_width + crop_w_h) // 2,
                       (img_height + crop_w_h) // 2))
    pil_img.save(savefile)


def resize_npy_save(save_dir, load_npy, resize_w_h=120):
    img = np.load(load_npy)
    half_size = int(img.shape[0] / 2)
    img = img[half_size - int(resize_w_h / 2):half_size + int(resize_w_h / 2),
          half_size - int(resize_w_h / 2):half_size + int(resize_w_h / 2)]
    np.save(save_dir, img)
