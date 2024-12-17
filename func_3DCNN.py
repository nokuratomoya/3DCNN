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

split_num = 1  # 分割サイズ　split_num*split_num分割される concatenateするときに使う
resize = 120  # resize * resize pixel
pixel = int(resize / split_num)
date = get_now()

data_num = 1200
train_data_num = int(data_num * 0.8)
value_data_num = data_num - train_data_num
spike_data_num = 50
# 刺激画像が始まる位置(教師画像は+1), emulator_25の場合でも200に設定
stim_head = 200

# データ拡張数
expansion_num = 1

# spike_data_name : "LNP" or "emulator" or "LNI" or "emulator_25"
spike_data_name = "emulator"

# 三次元出力用
output_3D = False
skip_data_num = 1
dataset_num_3D = int((2000 - spike_data_num) / skip_data_num)
# dataset_num_3D = 1400
start_num_3D = 642

# emulator_path = r'F:\train_data\20240108\0to1199'
emulator_path = r'H:\train_data\20240711\0to1199'
emulator_path_25 = r'H:\train_data\20240801\0to1199_2.5ms'
# dirname_main = r'F:\train_data\20240212\movie'
LNP_spike = r"H:\train_data\LNPspike\spikedata_original"
LNI_spike = r"H:\G\LNImodel\train_data\20240626\gain2_dt0.05\0to399"

if spike_data_name == "LNP":
    dirname_main = LNP_spike
elif spike_data_name == "emulator":
    dirname_main = emulator_path
elif spike_data_name == "LNI":
    dirname_main = LNI_spike
elif spike_data_name == "emulator_25":
    dirname_main = emulator_path_25

# dirname_main = r"F:\train_data\20240109\sustained\0to399"


def load_dataset():
    train_filename, value_filename = random_folder_select(dirname_main)
    # train_filename = natsorted(os.listdir(dirname_main))
    # value_filename = []
    train_filename = natsorted(train_filename)
    value_filename = natsorted(value_filename)
    print(train_filename)
    print(len(train_filename))
    print(value_filename)
    print(len(value_filename))
    # ---訓練データ(x_train)読み込み---
    start = time.time()

    if output_3D:
        x_trains, y_trains = load_data_3D(train_filename)
    else:
        # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
        x_trains, y_trains = load_data(train_filename)
    end = time.time()
    print("load_data_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    # ---訓練データ(x_train)リサイズ---
    start = time.time()
    # x_trains_resize: [train_data_num, spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
    x_trains_resize_list = x_trains_resize(x_trains)
    y_trains_resize_list = y_trains_resize(y_trains)
    end = time.time()
    print("resize_time:{}".format(end - start))
    # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
    # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")

    if split_num != 1:
        x_trains_resize_list, y_trains_resize_list = data_split(x_trains_resize_list, y_trains_resize_list)

    if expansion_num != 1:
        # ---訓練データ(x_train)拡張---
        start = time.time()
        x_trains_expansion_list, y_trains_expansion_list = data_expansion_main(x_trains_resize_list,
                                                                               y_trains_resize_list)
        end = time.time()
        print("expansion_time:{}".format(end - start))
        return x_trains_expansion_list, y_trains_expansion_list, value_filename

    # if output_3D:
    #     # ---訓練データ(x_train)3D化---
    #     start = time.time()
    #     x_trains_3D_list, y_trains_3D_list = data_3D_main(x_trains_resize_list, y_trains_resize_list)
    #     end = time.time()
    #     print("3D_time:{}".format(end - start))
    #     return x_trains_3D_list, y_trains_3D_list, value_filename

    # return x_trains_expansion_list, y_trains_expansion_list, value_filename
    return x_trains_resize_list, y_trains_resize_list, value_filename


def random_folder_select(dirname_main):
    # img0 -> trainに使用するファイル
    # img1 -> valueに使用するファイル
    used_filename = []

    all_filename = natsorted(os.listdir(dirname_main))
    filename = all_filename[:data_num]

    used_filename.append(filename[0])
    while len(used_filename) != train_data_num:

        list_temp = random.choice(filename)
        if list_temp in used_filename:
            continue
        elif list_temp == filename[1]:
            continue
        else:
            used_filename.append(list_temp)

    unused_filename = set(filename) ^ set(used_filename)
    if len(unused_filename) != value_data_num:
        print("value_data_num is not correct")
        exit()

    return natsorted(used_filename), natsorted(list(unused_filename))


def load_data(filename):
    # spikeファイル読み込み:X_train
    spike_data_list = []
    teacher_data_list = []
    count = 0
    for file in filename:
        # 読み込むファイルpath例：F:\train_data\20231128\stim400_cycle800ms\img0\img0

        # LNPspikeの場合
        if spike_data_name == "LNP":
            # LNPspikeの場合
            load_spike_path = LNP_spike + "\\spike_" + file + "\\"
            load_npy_path = list(
                map(lambda x: load_spike_path + "\\spike_" + str(x + stim_head) + ".npy", np.arange(spike_data_num)))
            # print(load_npy_path)

        elif spike_data_name == "emulator_25":
            load_spike_path = os.path.join(emulator_path_25, file, "img1_5")
            load_npy_path = [
                os.path.join(load_spike_path, f"img1_{stim_head + i}_{j}.npy")
                for i in range(int(spike_data_num / 2))
                for j in range(2)
            ]
        # print(load_spike_path)
        # 読み込むファイル名例：201~400
        # emulator,LNIの場合
        else:
            load_spike_path = os.path.join(dirname_main, file, "img1")
            load_npy_path = list(
                map(lambda x: load_spike_path + "\\img1_" + str(x + stim_head) + ".npy", np.arange(spike_data_num)))

        spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
        spike_data_list.append(spike_data_temp)

        y_path = os.path.join(emulator_path, file, "img0")
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


def load_data_3D(filename):
    # spikeファイル読み込み:X_train
    spike_data_list = []
    teacher_data_list = []
    count = 0
    for file in filename:
        print(file)
        for i in range(dataset_num_3D + start_num_3D):
            if i < start_num_3D:
                continue
            # 読み込むファイルpath例：F:\train_data\20231128\stim400_cycle800ms\img0\img0
            load_spike_path = os.path.join(dirname_main, file, "img1")
            # print(load_spike_path)
            # 読み込むファイル名例：201~400
            load_npy_path = list(
                map(lambda x: load_spike_path + "\\img1_" + str(x + i * skip_data_num) + ".npy",
                    np.arange(spike_data_num)))
            spike_data_temp = list(map(lambda x: np.load(x), load_npy_path))
            spike_data_list.append(spike_data_temp)

            y_path = os.path.join(dirname_main, file, "img0")
            load_y_path = y_path + "\\img0_" + str(spike_data_num + i * skip_data_num) + ".npy"
            teacher_data_list.append(np.load(load_y_path))

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


def data_3D_main(x_trains, y_trains):
    # x_trains:[train_data_num, spike_data_num, 120, 120]
    # y_trains:[train_data_num, 120, 120]
    # x_trains_3D_list: [train_data_num, spike_data_num, 120, 120, 1]
    # y_trains_3D_list: [train_data_num, 120, 120, 1]
    # データ拡張
    x_trains_3D_list = []
    y_trains_3D_list = []
    for i in range(len(x_trains)):
        x_temp, y_temp = data_3D_func(x_trains[i], y_trains[i])
        x_trains_3D_list.append(x_temp)
        y_trains_3D_list.append(y_temp)

    print("data_3D_end")
    print(
        f"x_trains_3D:{len(x_trains_3D_list)}*{len(x_trains_3D_list[0])}*{len(x_trains_3D_list[0][0])}*{len(x_trains_3D_list[0][0][0])}*{len(x_trains_3D_list[0][0][0][0])}")
    print(
        f"y_trains_3D:{len(y_trains_3D_list)}*{len(y_trains_3D_list[0])}*{len(y_trains_3D_list[0][0])}*{len(y_trains_3D_list[0][0][0])}")

    return x_trains_3D_list, y_trains_3D_list


def data_3D_func(x_train, y_train):
    x_trains1 = []
    y_trains1 = []

    # データ分割
    for i in range(spike_data_num):
        x_trains1.append(x_train[i])

    y_trains1.append(y_train)

    return x_trains1, y_trains1


def data_split(x_trains_ori, y_trains_ori):
    x_trains_split = []
    y_trains_split = []
    for i in range(len(x_trains_ori)):
        x_train = x_trains_ori[i]
        y_train = y_trains_ori[i]
        x_trains = []
        y_trains = []

        # ---分割プログラム---
        y_train_temp = np.split(y_train, split_num, 1)
        for i in range(split_num):
            y_trains.append(np.split(y_train_temp[i], split_num))

        for i in range(len(x_train)):
            x_train_temp = np.split(x_train[i], split_num, 1)
            for j in range(split_num):
                x_trains.append(np.split(x_train_temp[j], split_num))

        # ---平坦化---
        x_trains = list(itertools.chain.from_iterable(x_trains))
        y_trains = list(itertools.chain.from_iterable(y_trains))

        # ---並び替え---
        loop_num = split_num * split_num
        x_temp2 = []
        for i in range(loop_num):
            x_temp = []
            for j in range(i, len(x_trains), loop_num):
                x_temp.append(x_trains[j])
            x_temp2.append(x_temp)
        x_trains = x_temp2
        x_trains_split.append(x_trains)
        y_trains_split.append(y_trains)

    # shapeを変更 (32, split_num*split_num, 120, 120, 1) -> (32*split_num*split_num, 120, 120, 1)
    x_trains_split = np.array(x_trains_split)
    y_trains_split = np.array(y_trains_split)
    x_trains_split = x_trains_split.reshape(len(x_trains_split) * split_num * split_num, spike_data_num, pixel, pixel,
                                            1)
    y_trains_split = y_trains_split.reshape(len(y_trains_split) * split_num * split_num, pixel, pixel, 1)

    # print(f"x_trains_split:{len(x_trains_split)}*{len(x_trains_split[0])}*{len(x_trains_split[0][0])}*{len(x_trains_split[0][0][0])}*{len(x_trains_split[0][0][0][0])}")
    # print(f"y_trains_split:{len(y_trains_split)}*{len(y_trains_split[0])}*{len(y_trains_split[0][0])}*{len(y_trains_split[0][0][0])}")

    return x_trains_split, y_trains_split


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


def y_trains_save(y_trains, save_path):
    y_trains = y_trains.reshape(len(y_trains), pixel, pixel)
    y_trains *= 255
    for i in range(len(y_trains)):
        img_show(y_trains[i], save_path + "\\y_train\\" + str(i))

#
# def load_dataset_predict(filename):
#     # train_filename = natsorted(train_filename)
#     # value_filename = natsorted(value_filename)
#     # print(train_filename)
#     # print(value_filename)
#
#     # ---訓練データ(x_train)読み込み---
#     start = time.time()
#     # x_trains: [train_data_num, spike_data_num, 128, 128], y_trains: [train_data_num, 128, 128]
#     x_trains, y_trains = load_data(filename)
#     end = time.time()
#     # print("load_data_time:{}".format(end - start))
#     # print(f"x_trains:{len(x_trains)}*{len(x_trains[0])}*{len(x_trains[0][0])}*{len(x_trains[0][0][0])}")
#     # print(f"y_trains:{len(y_trains)}*{len(y_trains[0])}*{len(y_trains[0][0])}")
#
#     # ---訓練データ(x_train)リサイズ---
#     start = time.time()
#     # x_trains_resize: [train_data_num*spike_data_num, 120, 120], y_trains_resize: [train_data_num, 120, 120]
#     x_trains_resize_list = x_trains_resize(x_trains)
#     y_trains_resize_list = y_trains_resize(y_trains)
#     end = time.time()
#     # print("resize_time:{}".format(end - start))
#
#     # ---訓練データ(x_train)拡張---
#     # start = time.time()
#     x_trains_expansion_list, y_trains_expansion_list = data_expansion_main(x_trains_resize_list, y_trains_resize_list)
#     # end = time.time()
#     # print("expansion_time:{}".format(end - start))
#
#     return x_trains_expansion_list, y_trains_expansion_list
