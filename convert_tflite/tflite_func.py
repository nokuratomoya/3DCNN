import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Activation, AveragePooling3D
from PIL import Image
import pandas as pd
import numpy as np
import itertools
import os
import math
import random
import pickle
import csv
from natsort import natsorted
import multiprocessing
import time

from global_value import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


resize = 120  # resize * resize pixel
# pixel = int(resize / split_num)
pixel = pixel_size
# date = "20230125"
# date = get_now()
# train_data_date = "20230120"
train_data_num = dataset_num
value_data_num = 400 - train_data_num
spike_data_num = time_size
# 刺激画像が始まる位置(教師画像は+1)
stim_head = 201


def load_dataset_predict(filename):
    # ---訓練データ(x_train)読み込み---
    # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
    x_trains, y_trains = load_data(filename)
    # print("load_data_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    # x_trains_resize: [train_data_num*spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
    x_trains_resize_list = x_trains_resize(x_trains)
    x_trains_resize_list = np.array(x_trains_resize_list)
    # print(f"x_trains_resize_list:{np.shape(x_trains_resize_list)}")

    x_trains_resize_list = np.reshape(x_trains_resize_list, (
        x_trains_resize_list.shape[1], x_trains_resize_list.shape[2], x_trains_resize_list.shape[3]))

    # ---訓練データの(split_num*split_num)分割---
    # x_trains_split: [train_data_num*spike_data_num, split_num * split_num, pixel, pixel]
    # y_trains_split: [train_data_num, split_num * split_num, pixel, pixel]
    # x_trains_split_list, y_trains_split_list = data_split(x_trains_resize_list, y_trains_resize_list)
    x_trains_split_list = data_split(x_trains_resize_list)

    # print(f"x_trains_split_list:{np.shape(x_trains_resize_list)}")
    return x_trains_split_list


# 1まいだけ
def load_data(filename):
    # spikeファイル読み込み:X_train
    spike_data_list = []
    teacher_data_list = []
    count = 0

    # 読み込むファイルpath例：F:\train_data\20231128\stim400_cycle800ms\img0\img0
    load_spike_path = os.path.join(dirname_main, filename, "img1")
    # print(load_spike_path)
    # 読み込むファイル名例：201~400
    load_npy_path = list(
        map(lambda x: load_spike_path + "\\img1_" + str(x + stim_head) + ".npy", np.arange(spike_data_num)))
    spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
    spike_data_list.append(spike_data_temp)

    y_path = os.path.join(dirname_main, filename, "img0")
    load_y_path = y_path + "\\img0_" + str(stim_head + 1) + ".npy"
    teacher_data_list.append(np.load(load_y_path))

    # print("load_data_end")
    # print(
    #     f"x_trains:{len(spike_data_list)}*{len(spike_data_list[0])}*{len(spike_data_list[0][0])}*{len(spike_data_list[0][0][0])}")
    # print(f"y_trains:{len(teacher_data_list)}*{len(teacher_data_list[0])}*{len(teacher_data_list[0][0])}")

    return spike_data_list, teacher_data_list


def load_dataset_predict_3D(filename, i):
    # ---訓練データ(x_train)読み込み---
    # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
    # x_trains, y_trains = load_data_3D(filename, i)
    x_trains = load_data_3D(filename, i)
    # print("load_data_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    # x_trains_resize: [train_data_num*spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
    x_trains_resize_list = x_trains_resize(x_trains)
    # y_trains_resize_list = y_trains_resize(y_trains)
    x_trains_resize_list = np.array(x_trains_resize_list)
    # y_trains_resize_list = np.array(y_trains_resize_list)

    # x_trains_resize_list = np.reshape(x_trains_resize_list, (
    #     x_trains_resize_list.shape[1], x_trains_resize_list.shape[2], x_trains_resize_list.shape[3]))


    # ---訓練データの(split_num*split_num)分割---
    # x_trains_split: [train_data_num*spike_data_num, split_num * split_num, pixel, pixel]
    # y_trains_split: [train_data_num, split_num * split_num, pixel, pixel]
    # x_trains_split_list = data_split(x_trains_resize_list)

    return x_trains_resize_list


def load_data_3D(filename, i):
    # spikeファイル読み込み:X_train
    spike_data_list = []
    teacher_data_list = []
    count = 0

    # 読み込むファイルpath例：F:\train_data\20231128\stim400_cycle800ms\img0\img0
    load_spike_path = os.path.join(dirname_main, filename, "img1")
    # print(load_spike_path)
    # 読み込むファイル名例：201~400
    load_npy_path = list(
        map(lambda x: load_spike_path + "\\img1_" + str(x + i) + ".npy", np.arange(spike_data_num)))
    spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
    spike_data_list.append(spike_data_temp)

    # y_path = os.path.join(dirname_main, filename, "img0")
    # load_y_path = y_path + "\\img0_" + str(stim_head + 1) + ".npy"
    # teacher_data_list.append(np.load(load_y_path))

    # print("load_data_end")
    # print(
    #     f"x_trains:{len(spike_data_list)}*{len(spike_data_list[0])}*{len(spike_data_list[0][0])}*{len(spike_data_list[0][0][0])}")
    # print(f"y_trains:{len(teacher_data_list)}*{len(teacher_data_list[0])}*{len(teacher_data_list[0][0])}")

    return spike_data_list


def load_npy(files):
    temp = []
    for file in files:
        temp.append(np.load(file))
    return temp
    # マルチプロセス
    # load_npy_list = np.load(files)
    # return load_npy_list


def x_trains_resize(x_trains):
    # x_trains:[train_data_num, spike_data_num, 128, 128]
    # x_trains_resize:[train_data_num*spike_data_num, 120, 120]
    x_trains_resize_list = []

    half_size = int(len(x_trains[0][0]) / 2)
    for i in range(len(x_trains)):
        temp_list = []
        for j in range(len(x_trains[i])):
            temp = x_trains[i][j][(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
                   (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]
            temp_list.append(temp)
        x_trains_resize_list.append(temp_list)
    # print("max:{}".format(max))

    # print("x_trains_resize_end")
    # print(
    #     f"x_trains:{len(x_trains_resize_list)}*{len(x_trains_resize_list[0])}*{len(x_trains_resize_list[0][0])}*{len(x_trains_resize_list[0][0][0])}")

    return x_trains_resize_list


def y_trains_resize(y_trains):
    # y_trains: [train_data_num, 128, 128]
    # y_trains_resize: [train_data_num, 120, 120]
    y_trains_resize_list = []
    half_size = int(len(y_trains[0]) / 2)
    temp = []
    for i in range(len(y_trains)):
        temp = y_trains[i][(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
               (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]
        y_trains_resize_list.append(temp)

    # print("y_trains_resize_end")
    # print(f"y_trains:{len(y_trains_resize_list)}*{len(y_trains_resize_list[0])}*{len(y_trains_resize_list[0][0])}")

    return y_trains_resize_list


def data_expansion_main(x_trains, y_trains):
    # x_trains:[train_data_num*spike_data_num, 120, 120]
    # y_trains:[train_data_num, 120, 120]
    # x_trains_expansion_list: [train_data_num, spike_data_num * expansion_num, 120, 120]
    # y_trains_expansion_list: [train_data_num, expansion_num, 120, 120]
    # データ拡張
    x_trains_expansion_list = []
    y_trains_expansion_list = []
    for i in range(len(x_trains)):
        x_temp, y_temp = data_expansion_func(x_trains[i], y_trains[i])
        x_trains_expansion_list.append(x_temp)
        y_trains_expansion_list.append(y_temp)

    # 平坦化
    x_trains_expansion_list = list(itertools.chain.from_iterable(x_trains_expansion_list))
    y_trains_expansion_list = list(itertools.chain.from_iterable(y_trains_expansion_list))

    print("data_expansion_end")
    print(
        f"x_trains_expansion:{len(x_trains_expansion_list)}*{len(x_trains_expansion_list[0])}*{len(x_trains_expansion_list[0][0])}*{len(x_trains_expansion_list[0][0][0])}")
    print(
        f"y_trains_expansion:{len(y_trains_expansion_list)}*{len(y_trains_expansion_list[0])}*{len(y_trains_expansion_list[0][0])}")

    return x_trains_expansion_list, y_trains_expansion_list


def data_expansion_func(x_train, y_train):
    x_trains1 = []
    y_trains1 = []
    expansion_num = 10
    """
    # ランダムに行、列の生成(かぶりなし)
    while len(col_all) != expansion_num:
        temp_row = random.randint(0, resize - pixel - 1)
        temp_col = random.randint(0, resize - pixel - 1)
        if temp_col not in col_all or temp_row not in row_all:
            row_all.append(temp_row)
            col_all.append(temp_col)

    """
    # データ分割
    for i in range(expansion_num):
        x_trains_temp = []
        row = random.randint(0, resize - pixel - 1)
        col = random.randint(0, resize - pixel - 1)
        y_trains1.append(y_train[row:(row + pixel), col:(col + pixel)])
        for j in range(spike_data_num):
            x_trains_temp.append(x_train[j][row:(row + pixel), col:(col + pixel)])

        x_trains1.append(x_trains_temp)

    return x_trains1, y_trains1


def data_split(x_train):
    x_trains = []

    # ---分割プログラム---
    # y_train_temp = np.split(y_train, split_num, 1)
    # for i in range(split_num):
    # y_trains.append(np.split(y_train_temp[i], split_num))

    for i in range(len(x_train)):
        x_train_temp = np.split(x_train[i], split_num, 1)
        for j in range(split_num):
            x_trains.append(np.split(x_train_temp[j], split_num))

    # ---平坦化---
    x_trains = list(itertools.chain.from_iterable(x_trains))
    # y_trains = list(itertools.chain.from_iterable(y_trains))

    # ---並び替え---
    loop_num = split_num * split_num
    x_temp2 = []
    for i in range(loop_num):
        x_temp = []
        for j in range(i, len(x_trains), loop_num):
            x_temp.append(x_trains[j])
        x_temp2.append(x_temp)
    x_trains = x_temp2

    # print("data_split_end")
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    return x_trains


def plot_history(history, path):
    plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(hspace=0.3)

    """
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], "-", label="accuracy")
    # plt.plot(history.history['val_accuracy'], "-", label="val_accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    """

    # plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], "-", label="loss")
    # plt.plot(history.history['val_loss'], "-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig(path)
    # plt.show()


def hist_csv_save(history, path):
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(path)


def save_data_txt(save_data, save_dir_name):
    f = open(save_dir_name, 'wb')
    pickle.dump(save_data, f)
    f.close()


# save_dataはlist型にする
def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)

        # データを追加して保存する
        writer.writerow(save_data)

    file.close()


def img_show(data, save_file):
    img = Image.fromarray(data)
    # img.show()
    img.convert('L').save(save_file + ".jpg")


def pre_npy_save(data, save_file):
    np.save(save_file, data)


def pre_concatenate(x_pre):
    for i in range(split_num):
        row_c = x_pre[4 * i]
        for j in range(split_num - 1):
            row_c = np.concatenate([row_c, x_pre[(4 * i) + j + 1]])

        if i == 0:
            col_c = row_c
        else:
            col_c = np.concatenate([col_c, row_c], 1)

    return col_c
