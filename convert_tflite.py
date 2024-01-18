import cv2
import glob
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from natsort import natsorted
import os
import random
import time

from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

split_num = 1  # 分割サイズ　split_num*split_num分割される concatenateするときに使う
resize = 120  # resize * resize pixel
# pixel = int(resize / split_num)
pixel = 120
# date = "20230125"
train_data_date = "20230120"
train_data_num = 320
value_data_num = 400 - train_data_num
spike_data_num = 100
# 刺激画像が始まる位置(教師画像は+1)
stim_head = 201
# データ拡張数
expansion_num = 1

# 三次元出力用
output_3D = False
skip_data_num = 23
dataset_num_3D = int((800 - spike_data_num) / skip_data_num)

dirname_main = r'F:\train_data\20231129\stim400_cycle800ms'

def main():
    model_save_path = r"C:\Users\AIlab\labo\3DCNN\results\20231205\result_kernel_100_3_3\model\\"
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_save_path + "model_dataset=320_e=2104_b=16.h5")
    model = load_model(model_save_path + "model_dataset=320_e=2104_b=16.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # x_trains = load_dataset()
    converter.representative_dataset = load_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                         tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    save_path = r"C:\Users\AIlab\labo\3DCNN\results\20231205\result_kernel_100_3_3\model\\"
    model_name = "tflite_model"
    open(save_path + model_name + '.tflite', 'wb').write(tflite_model)


def load_dataset():
    train_filename, value_filename = random_folder_select(dirname_main)
    # train_filename = natsorted(train_filename)
    # value_filename = natsorted(value_filename)
    print(train_filename)
    print(value_filename)

    # ---訓練データ(x_train)読み込み---
    start = time.time()

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

    x_trains = np.array(x_trains_resize_list)
    x_trains = x_trains[:, :, :, :, np.newaxis]
    x_trains = x_trains.astype('float32')
    # x_trains_t = np.expand_dims(x_trains, axis=0)

    yield [x_trains]


def random_folder_select(dirname_main):
    # img0 -> trainに使用するファイル
    # img1 -> valueに使用するファイル
    used_filename = []

    all_filename = natsorted(os.listdir(dirname_main))
    used_filename.append(all_filename[0])
    while len(used_filename) != train_data_num:

        list_temp = random.choice(all_filename)
        if list_temp in used_filename:
            continue
        elif list_temp == all_filename[1]:
            continue
        else:
            used_filename.append(list_temp)

    unused_filename = set(all_filename) ^ set(used_filename)

    return natsorted(used_filename), natsorted(list(unused_filename))


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


if __name__ == '__main__':
    main()
