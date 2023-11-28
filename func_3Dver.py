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
from global_value import get_now

split_num = 4  # 分割サイズ　split_num*split_num分割される
resize = 120  # resize * resize pixel
pixel = int(resize / split_num)
# date = "20230125"
date = get_now()
train_data_date = "20230120"
train_data_num = 40
dirname_main = r'train_data\\' + train_data_date


def load_dataset():
    spike_data_num = 1000  # スパイクデータの枚数(1枚:0.05ms)
    spike_range_in = 10  # 変えない
    spike_range = int(spike_data_num / spike_range_in)  # 最終的なスパイクデータ数

    # ---訓練データ(x_train)読み込み---
    x_trains = []
    y_trains = []

    used_filename, unused_filename = random_folder_select(dirname_main)

    for dirname in used_filename:
        """
        if dirname == 'adler_data':  # debug
            continue
        """
        dirname_x = dirname_main + '\\' + dirname + '\img1'
        dirname_y = dirname_main + '\\' + dirname + '\img0'

        x_train = []
        for i in range(spike_range):  # 100枚のスパイクデータ
            spike_data = []
            for j in range(spike_range_in):
                filename_x = '\\img1_' + str(i) + '_' + str(j) + '.csv'

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

        x_train_temp, y_train_temp = data_split(x_train, y_train)
        x_trains.append(x_train_temp)
        y_trains.append(y_train_temp)
        x_train_temp, y_train_temp = data_expansion(x_train, y_train)
        x_trains.append(x_train_temp)
        y_trains.append(y_train_temp)
        # print(f"x_trains={len(x_train_temp)},y_trains={len(y_train_temp)}")
        print(f"{dirname} is finished")

    x_trains = list(itertools.chain.from_iterable(x_trains))
    y_trains = list(itertools.chain.from_iterable(y_trains))

    print('使用したファイル : ' + str(used_filename))
    print('データセット数 : ' + str(len(x_trains)))
    return x_trains, y_trains, unused_filename


def load_data_predict(filename):
    spike_data_num = 1000  # スパイクデータの枚数(1枚:0.05ms)
    spike_range_in = 10  # 不変
    spike_range = int(spike_data_num / spike_range_in)  # 最終的なスパイクデータ数
    dirname_x = dirname_main + '\\' + filename + '\img1'
    dirname_y = dirname_main + '\\' + filename + '\img0'
    x_train = []

    # x_pre 読み込み
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

    # y_pre 読み込み
    filename_y = '\img0_0.csv'

    y_train = np.loadtxt(dirname_y + filename_y, delimiter=',')
    y_train = y_train[(half_size - (int(resize / 2))):(half_size + (int(resize / 2))),
              (half_size - (int(resize / 2))):(half_size + (int(resize / 2)))]

    x_train, y_train = data_split(x_train, y_train)

    # x_train = list(itertools.chain.from_iterable(x_train))
    # y_train = list(itertools.chain.from_iterable(y_train))

    return x_train, y_train


# スパイクデータ10枚を重ね合わせる
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


def data_split(x_train, y_train):
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

    return x_trains, y_trains


def data_expansion(x_train, y_train):
    x_trains1 = []
    y_trains1 = []
    expansion_num = 100
    row_all = []
    col_all = []
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
        for j in range(100):
            x_trains_temp.append(x_train[j][row:(row + pixel), col:(col + pixel)])

        x_trains1.append(x_trains_temp)

    return x_trains1, y_trains1


def data_expansion_all(x_train, y_train):
    x_trains1 = []
    y_trains1 = []
    row_all = []
    col_all = []
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
    for i in range(resize - pixel):
        for j in range(resize - pixel):
            x_trains_temp = []
            y_trains1.append(y_train[i:(i + pixel), j:(j + pixel)])
            for k in range(100):
                x_trains_temp.append(x_train[k][i:(i + pixel), j:(j + pixel)])

            x_trains1.append(x_trains_temp)

    return x_trains1, y_trains1


def img_show(data, save_file):
    img = Image.fromarray(data)
    # img.show()
    img.convert('L').save(save_file + "_one" + ".jpg")


def pre_csv_save(data, save_file):
    np.savetxt(save_file + ".csv", data, delimiter=",")


# 使ってない？
def img_shows(list, save_file):
    image = []
    num = len(list)  # リストの要素数
    row = int(math.sqrt(num))
    col = int(row)
    count = 1

    for i in range(num):
        image.append(Image.fromarray(list[i]))

    for i in range(row):
        for j in range(i, num, col):
            plt.subplot(col, row, count)
            plt.axis("off")
            plt.imshow(image[j])
            # image[j].convert('L').save(save_file + "_" + str(count) + ".jpg")
            count += 1

    plt.subplots_adjust()
    plt.savefig(save_file + "_total.jpg")
    plt.show()


# predict_imageだけで使う用
def img_compare(list, save_file, dirpath):
    image = []
    num = len(list)
    col = 2
    row = 1

    for i in range(num):
        image.append(Image.fromarray(list[i]))
        plt.subplot(row, col, i + 1)
        if i == 0:
            plt.title(save_file.replace(dirpath + 'pre_', ''))
        plt.axis("off")
        plt.imshow(image[i])
        # image[j].convert('L').save(save_file + "_" + str(count) + ".jpg")

    plt.subplots_adjust()
    # plt.title(save_file.replace(dirpath, ''))
    plt.savefig(save_file + "_compare.jpg")
    plt.show()


def img_MI(rec_img_data, rec_dir_name, save_dir_name):
    img_num = 2  # 表示する画像の枚数
    col = 2  # 列
    row = 1  # 行

    plt.figure(figsize=(8, 6))
    for i in range(img_num):
        img = Image.open(rec_dir_name[i])
        img = np.asarray(img)
        img = Image.fromarray(img)
        plt.subplot(row, col, i + 1)
        plt.title(rec_img_data[i])
        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.subplots_adjust()
    # plt.savefig(save_dir_name + "_compare_k100.jpg")
    # plt.show()


def model_build():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(10, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   , activation='relu'),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            Conv3D(filters=16, kernel_size=(10, 3, 3), padding='same', activation='relu'),
            # (100, 30, 30, 1) => (100, 30, 30, 1)
            MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=32, kernel_size=(5, 3, 3), padding='same', activation='relu'),
            MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=(1, 3, 3), padding='same', activation='sigmoid'),
            # Flatten(),  # (57600)
            # Dense(128, activation='relu'),  #
            # Dropout(0.25),
            # Dense(classes, activation='sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_build_kernel_20():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(20, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   , activation='relu'),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            Conv3D(filters=16, kernel_size=(20, 3, 3), padding='same', activation='relu'),
            # (100, 30, 30, 1) => (100, 30, 30, 1)
            MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=32, kernel_size=(5, 3, 3), padding='same', activation='relu'),
            MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=(1, 3, 3), padding='same', activation='sigmoid'),
            # Flatten(),  # (57600)
            # Dense(128, activation='relu'),  #
            # Dropout(0.25),
            # Dense(classes, activation='sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_build_kernel_100():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   , activation='relu'),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same', activation='relu'),
            # (100, 30, 30, 1) => (100, 30, 30, 1)
            MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=32, kernel_size=(25, 3, 3), padding='same', activation='relu'),
            MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=64, kernel_size=(5, 3, 3), padding='same', activation='relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=(1, 3, 3), padding='same', activation='sigmoid'),
            # Flatten(),  # (57600)
            # Dense(128, activation='relu'),  #
            # Dropout(0.25),
            # Dense(classes, activation='sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_build_kernel_100_less_old():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   ),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            # BatchNormalization(),
            Activation('relu'),
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=32, kernel_size=(25, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=64, kernel_size=(5, 3, 3), padding='same'),
            Activation('relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=(1, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_build_kernel_100_less():
    model = Sequential(
        layers=[
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same', input_shape=(100, 30, 30, 1)
                   ),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            # BatchNormalization(),
            Activation('relu'),
            Conv3D(filters=16, kernel_size=(100, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=32, kernel_size=(25, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=64, kernel_size=(5, 3, 3), padding='same'),
            Activation('relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=(1, 3, 3), padding='same'),
            # BatchNormalization(),
            Activation('sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_build_kernel_change(kernel_time, kernel_xy, filter_size):
    kernel_size1 = [kernel_time, kernel_xy, kernel_xy]

    # [5, 10, 20]
    if kernel_time <= 20:
        kernel_size2 = [int(kernel_time / 2), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 4), kernel_xy, kernel_xy]
    # [40, 60, 80]
    elif kernel_time <= 80:
        kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 16), kernel_xy, kernel_xy]
    # [100]
    else:
        kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 20), kernel_xy, kernel_xy]

    kernel_size4 = [1, kernel_xy, kernel_xy]

    print(kernel_size1)
    model = Sequential(
        layers=[

            Conv3D(filters=filter_size, kernel_size=kernel_size1, kernel_initializer='lecun_uniform', padding='same',
                   input_shape=(100, 30, 30, 1)
                   ),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            # BatchNormalization(),
            Activation('relu'),
            Conv3D(filters=filter_size, kernel_size=kernel_size1, padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            AveragePooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=filter_size * 2, kernel_size=kernel_size2, padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            AveragePooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=filter_size * 4, kernel_size=kernel_size3, padding='same'),
            Activation('relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            AveragePooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)
            # MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=kernel_size4, padding='same'),
            # BatchNormalization(),
            Activation('sigmoid')

        ]
    )

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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


def crop_img(savefile, path, crop_w_h):
    im = Image.open(path)
    img_width, img_height = im.size
    pil_img = im.crop(((img_width - crop_w_h) // 2,
                       (img_height - crop_w_h) // 2,
                       (img_width + crop_w_h) // 2,
                       (img_height + crop_w_h) // 2))
    pil_img.save(savefile)


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
