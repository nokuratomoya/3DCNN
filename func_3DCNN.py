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
# from global_value import get_now


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

split_num = 4  # 分割サイズ　split_num*split_num分割される
resize = 120  # resize * resize pixel
pixel = int(resize / split_num)
# date = "20230125"
# date = get_now()
train_data_date = "20230120"
train_data_num = 320
value_data_num = 400 - train_data_num
spike_data_num = 100
# 刺激画像が始まる位置
stim_head = 201
dirname_main = r'F:\train_data\20231129\stim400_cycle800ms'


def main():
    train_filename, value_filename = random_folder_select(dirname_main)
    train_filename = natsorted(train_filename)
    value_filename = natsorted(value_filename)
    print(train_filename)
    print(value_filename)

    # ---訓練データ(x_train)読み込み---
    start = time.time()
    # x_trains: [train_data_num*spike_data_num*128*128], y_trains: [train_data_num*128*128]
    x_trains, y_trains = load_data(train_filename)
    end = time.time()
    print("load_data_time:{}".format(end - start))
    print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    start = time.time()
    # x_trains_resize: [train_data_num*spike_data_num*120*120], y_trains_resize: [train_data_num*120*120]
    x_trains_resize_list = x_trains_resize(x_trains)
    y_trains_resize_list = y_trains_resize(y_trains)
    end = time.time()
    print("resize_time:{}".format(end - start))
    print(f"x_trains:{len(x_trains_resize_list)}*{len(x_trains_resize_list[0])}*{len(x_trains_resize_list[0][0])}*{len(x_trains_resize_list[0][0][0])}")
    print(f"y_trains:{len(y_trains_resize_list)}*{len(y_trains_resize_list[0])}*{len(y_trains_resize_list[0][0])}")


    # ---訓練データ(x_train)分割---


def x_trains_resize(x_trains):
    # x_trains: [train_data_num*spike_data_num*128*128]
    # x_trains_resize: [train_data_num*spike_data_num*120*120]
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
    return x_trains_resize_list


def y_trains_resize(y_trains):
    # y_trains: [train_data_num*128*128]
    # y_trains_resize: [train_data_num*120*120]
    y_trains_resize_list = []
    half_size = int(len(y_trains[0]) / 2)
    temp = []
    for i in range(len(y_trains)):
        temp = y_trains[i][(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
                     (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]
        y_trains_resize_list.append(temp)
    return y_trains_resize_list


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
        load_npy_path = list(map(lambda x: load_spike_path + "\\img1_" + str(x + stim_head) + ".npy", np.arange(spike_data_num)))
        spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
        spike_data_list.append(spike_data_temp)

        y_path = os.path.join(dirname_main, file, "img0")
        load_y_path = y_path + "\\img0_" + str(stim_head + 1) + ".npy"
        teacher_data_list.append(np.load(load_y_path))

        # count += 1
        # if count == 10:
        #     break

    return spike_data_list, teacher_data_list


def load_npy(files):
    temp = []
    for file in files:
        temp.append(np.load(file))
    return temp
    # マルチプロセス
    # load_npy_list = np.load(files)
    # return load_npy_list


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


if __name__ == "__main__":
    main()
