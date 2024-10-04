import csv
from natsort import natsorted
import os
import numpy as np
import pickle

date = "20240126_2"
train_data_num = 64
epochs = 1733
batch_size = 16
time_kernel = 100
xy_kernel = 3

dataset_num = 1200

train_filename = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\result_kernel_" + str(time_kernel) + "_" + str(
    xy_kernel) + "_" + str(xy_kernel) + r"\model\train_data=" + str(train_data_num) + "_e=" + str(epochs) + "_b=" + str(
    batch_size)
test_filename = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\result_kernel_" + str(time_kernel) + "_" + str(
    xy_kernel) + "_" + str(xy_kernel) + r"\model\test_data=" + str(train_data_num) + "_e=" + str(epochs) + "_b=" + str(
    batch_size)
train_dir = r"F:\train_data\20240108\0to1199"

save = "test"


def main():
    if save == "train":
        f = open(test_filename + ".txt", 'rb')
    else:
        f = open(train_filename + ".txt", 'rb')
    unused_filename = natsorted(pickle.load(f))
    f.close()
    all_filename = natsorted(os.listdir(train_dir))
    all_filename = all_filename[:dataset_num]
    used_filename = set(all_filename) ^ set(unused_filename)
    used_filename = natsorted(used_filename)
    print("used_filename:" + str(len(used_filename)))
    print("unused_filename:" + str(len(unused_filename)))

    if save == "train":
        save_data_csv(used_filename, train_filename + "_1.csv")
        save_data_txt(used_filename, train_filename + "_1.txt")
    else:
        save_data_csv(used_filename, test_filename + "_1.csv")
        save_data_txt(used_filename, test_filename + "_1.txt")


def save_data_txt(save_data, save_dir_name):
    f = open(save_dir_name, 'wb')
    pickle.dump(save_data, f)
    f.close()


def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # データを追加して保存する
        writer.writerow(save_data)
    file.close()


if __name__ == "__main__":
    main()
