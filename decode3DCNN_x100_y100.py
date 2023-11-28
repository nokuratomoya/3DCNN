from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pathlib
import os
import time
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Dropout, BatchNormalization, Activation
from PIL import Image
import pandas as pd
import numpy as np
import itertools
import os
import csv
import random
from global_value import get_now

EPOCHS = 10000
BATCH_SIZE = 16
split_num = 4  # 分割サイズ　split_num*split_num分割される
resize = 120  # resize * resize pixel
pixel = int(resize / split_num)
# date = "20230125"
date = get_now()
train_data_date = "20230120"
train_data_num = 40
dirname_main = r'train_data\\' + train_data_date


# BATCH_SIZE_ALL = [8, 16, 32]

def main():
    # dataset準備
    start_load_time = time.perf_counter()
    x_trains, y_trains, unused_filename = load_dataset_y100()  # x_trains = [:, 100, 30, 30],y_trains = [:, 30, 30]
    end_load_time = time.perf_counter()
    print(f"load_time:{end_load_time - start_load_time}s")
    dataset_num = len(x_trains)
    # reshape
    x_trains, y_trains = np.array(x_trains), np.array(y_trains)
    x_trains, y_trains = x_trains[:, :, :, :, np.newaxis], y_trains[:, :, :, :, np.newaxis]  # np.reshape
    y_trains /= 255.0
    x_trains, y_trains = x_trains.astype('float32'), y_trains.astype('float32')
    #
    print(np.shape(x_trains), np.shape(y_trains))
    start_train_time = time.perf_counter()

    model_kernel_y100(x_trains, y_trains, unused_filename, dataset_num)


def load_dataset_y100():
    spike_data_num = 1000  # スパイクデータの枚数(1枚:0.05ms)
    spike_range_in = 10  # 変えない
    spike_range = int(spike_data_num / spike_range_in)  # 最終的なスパイクデータ数

    x_trains = []
    y_trains = []

    used_filename, unused_filename = random_folder_select(dirname_main)

    for dirname in used_filename:

        dirname_x = dirname_main + '\\' + dirname + '\img1'
        dirname_y = dirname_main + '\\' + dirname + '\img0'

        # ---訓練データ(x_train)読み込み---
        x_train = []
        for i in range(spike_range):  # 100枚のスパイクデータ
            spike_data = []
            for j in range(spike_range_in):
                filename_x = '\img1_' + str(i) + '_' + str(j) + '.csv'

                spike_temp = np.loadtxt(dirname_x + filename_x, delimiter=',')

                spike_data = spike_add(j, spike_temp, spike_data)

            half_size = int(spike_data.shape[0] / 2)
            spike_data = spike_data[(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
                         (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]
            x_train.append(spike_data)

        # ---正解ラベル(y_train)読み込み---
        filename_y = '\img0_0.csv'

        y_train = np.loadtxt(dirname_y + filename_y, delimiter=',')
        y_train = y_train[(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
                  (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]

        y_train = calc_y_csv(y_train)

        x_train_temp = data_split(x_train)
        y_train_temp = data_split(y_train)
        x_trains.append(x_train_temp)
        y_trains.append(y_train_temp)
        x_train_temp, y_train_temp = data_expansion(x_train, y_train)
        x_trains.append(x_train_temp)
        y_trains.append(y_train_temp)

        print(f"{dirname} is finished")

    x_trains = list(itertools.chain.from_iterable(x_trains))
    y_trains = list(itertools.chain.from_iterable(y_trains))

    print('使用したファイル : ' + str(used_filename))
    print('データセット数 : ' + str(len(x_trains) / 100))
    return x_trains, y_trains, unused_filename


def spike_add(j, spike_temp, spike_data):
    if j == 0:
        spike_data = np.array(spike_temp)

    else:
        spike_data = spike_data + spike_temp

    return spike_data


def random_folder_select(dirname_main):
    used_filename = []
    all_filename = os.listdir(dirname_main)
    while len(used_filename) != train_data_num:
        list_temp = random.choice(all_filename)
        if list_temp in used_filename:
            continue
        else:
            used_filename.append(list_temp)

    unused_filename = set(all_filename) ^ set(used_filename)

    return used_filename, list(unused_filename)


def calc_y_csv(y_train):
    # csvファイル読み込み
    load_dir = r"C:\Users\AIlab\labo\3DCNN\\" + "20230201" + r"\result\filter_weight\layer1\center\fft"
    load_csv = "inverse_fft.csv"
    with open(load_dir + '\\' + load_csv, 'r') as file:
        # csv.readerを使用してファイルを読み込む
        csv_reader = csv.reader(file)
        # 各行を処理する
        for row in csv_reader:
            # 行のデータを使用して何らかの処理を行う
            ifft_data = row

    ifft_data = np.asarray(ifft_data, dtype=float)

    # 正規化
    ifft_min_max = (ifft_data - ifft_data.min()) / (ifft_data.max() - ifft_data.min())

    y_train_stock = []
    for i in range(100):
        calc_data = y_train * ifft_min_max[i]
        y_train_stock.append(calc_data)

    return y_train_stock


def data_split(train):
    trains = []

    # ---分割プログラム---
    for i in range(len(train)):
        x_train_temp = np.split(train[i], split_num, 1)
        for j in range(split_num):
            trains.append(np.split(x_train_temp[j], split_num))

    # ---平坦化---
    trains = list(itertools.chain.from_iterable(trains))

    # ---並び替え---
    loop_num = split_num * split_num
    temp2 = []
    for i in range(loop_num):
        temp = []
        for j in range(i, len(trains), loop_num):
            temp.append(trains[j])
        temp2.append(temp)
    trains = temp2

    return trains


def data_expansion(x_train, y_train):
    x_trains1 = []
    y_trains1 = []
    expansion_num = 100

    # データ分割
    for i in range(expansion_num):
        x_trains_temp = []
        y_trains_temp = []
        row = random.randint(0, resize - pixel - 1)
        col = random.randint(0, resize - pixel - 1)

        for j in range(100):
            x_trains_temp.append(x_train[j][row:(row + pixel), col:(col + pixel)])
            y_trains_temp.append(y_train[j][row:(row + pixel), col:(col + pixel)])

        x_trains1.append(x_trains_temp)
        y_trains1.append(y_trains_temp)

    return x_trains1, y_trains1


def model_kernel_y100(x_trains, y_trains, unused_filename, dataset_num):
    start_train_time = time.perf_counter()
    model = model_build_y100()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + r'\result\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(save_unused_dir, 'wb')
    pickle.dump(unused_filename, f)
    f.close()

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    train_dir = "train_data\\20230120"
    all_filename = os.listdir(train_dir)
    used_filename = set(all_filename) ^ set(unused_filename)
    f = open(save_used_dir, 'wb')
    pickle.dump(used_filename, f)
    f.close()

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)


def model_build_y100():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   ),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            # BatchNormalization(),
            Activation('relu'),
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


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


if __name__ == '__main__':
    main()
