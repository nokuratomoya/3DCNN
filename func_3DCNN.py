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

from global_value import get_now

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

split_num = 4  # 分割サイズ　split_num*split_num分割される
resize = 120  # resize * resize pixel
# pixel = int(resize / split_num)
pixel = 120
# date = "20230125"
date = get_now()
train_data_date = "20230120"
train_data_num = 320
value_data_num = 400 - train_data_num
spike_data_num = 100
# 刺激画像が始まる位置(教師画像は+1)
stim_head = 201
# データ拡張数
expansion_num = 1

dirname_main = r'F:\train_data\20231129\stim400_cycle800ms'


def load_dataset():
    train_filename, value_filename = random_folder_select(dirname_main)
    train_filename = natsorted(train_filename)
    value_filename = natsorted(value_filename)
    print(train_filename)
    print(value_filename)

    # ---訓練データ(x_train)読み込み---
    start = time.time()
    # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
    x_trains, y_trains = load_data(train_filename)
    end = time.time()
    print("load_data_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    start = time.time()
    # x_trains_resize: [train_data_num*spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
    x_trains_resize_list = x_trains_resize(x_trains)
    y_trains_resize_list = y_trains_resize(y_trains)
    end = time.time()
    print("resize_time:{}".format(end - start))

    # ---訓練データ(x_train)拡張---
    start = time.time()
    # x_trains_expansion_list, y_trains_expansion_list = data_expansion_main(x_trains_resize_list, y_trains_resize_list)
    end = time.time()
    print("expansion_time:{}".format(end - start))

    # return x_trains_expansion_list, y_trains_expansion_list, value_filename
    return x_trains_resize_list, y_trains_resize_list, value_filename


def load_dataset_predict(filename):
    # train_filename = natsorted(train_filename)
    # value_filename = natsorted(value_filename)
    # print(train_filename)
    # print(value_filename)

    # ---訓練データ(x_train)読み込み---
    start = time.time()
    # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
    x_trains, y_trains = load_data(filename)
    end = time.time()
    # print("load_data_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    start = time.time()
    # x_trains_resize: [train_data_num*spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
    x_trains_resize_list = x_trains_resize(x_trains)
    y_trains_resize_list = y_trains_resize(y_trains)
    end = time.time()
    # print("resize_time:{}".format(end - start))

    # ---訓練データ(x_train)拡張---
    # start = time.time()
    x_trains_expansion_list, y_trains_expansion_list = data_expansion_main(x_trains_resize_list, y_trains_resize_list)
    # end = time.time()
    # print("expansion_time:{}".format(end - start))

    return x_trains_expansion_list, y_trains_expansion_list


def random_folder_select(dirname_main):
    used_filename = []
    all_filename = natsorted(os.listdir(dirname_main))
    while len(used_filename) != train_data_num:
        list_temp = random.choice(all_filename)
        if list_temp in used_filename:
            continue
        else:
            used_filename.append(list_temp)

    unused_filename = set(all_filename) ^ set(used_filename)

    return used_filename, list(unused_filename)


def load_data(filename):
    # spikeファイル読み込み:X_train
    spike_data_list = []
    teacher_data_list = []
    count = 0
    for file in filename:
        # 読み込むファイルpath例：F:\train_data\20231128\stim400_cycle800ms\img0\img0
        load_spike_path = os.path.join(dirname_main, file, "img1")
        print(load_spike_path)
        # 読み込むファイル名例：201~400
        load_npy_path = list(
            map(lambda x: load_spike_path + "\\img1_" + str(x + stim_head) + ".npy", np.arange(spike_data_num)))
        spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
        spike_data_list.append(spike_data_temp)

        y_path = os.path.join(dirname_main, file, "img0")
        load_y_path = y_path + "\\img0_" + str(stim_head + 1) + ".npy"
        teacher_data_list.append(np.load(load_y_path))

        # count += 1
        # if count == 10:
        #     break

    print("load_data_end")
    print(
        f"x_trains:{len(spike_data_list)}*{len(spike_data_list[0])}*{len(spike_data_list[0][0])}*{len(spike_data_list[0][0][0])}")
    print(f"y_trains:{len(teacher_data_list)}*{len(teacher_data_list[0])}*{len(teacher_data_list[0][0])}")

    return spike_data_list, teacher_data_list


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
    print("max:{}".format(max))

    print("x_trains_resize_end")
    print(
        f"x_trains:{len(x_trains_resize_list)}*{len(x_trains_resize_list[0])}*{len(x_trains_resize_list[0][0])}*{len(x_trains_resize_list[0][0][0])}")

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

    print("y_trains_resize_end")
    print(f"y_trains:{len(y_trains_resize_list)}*{len(y_trains_resize_list[0])}*{len(y_trains_resize_list[0][0])}")

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
    img.convert('L').save(save_file + "_one" + ".jpg")


def pre_csv_save(data, save_file):
    np.savetxt(save_file + ".csv", data, delimiter=",")


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
