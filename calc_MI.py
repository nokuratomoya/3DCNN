# unused_dataをpredictし相互情報量を求める

from tensorflow.keras.models import load_model
from func_3Dver import load_data_predict, split_num, img_show, img_compare, pre_concatenate, crop_img
import numpy as np
from PIL import Image
import os
import pickle
import csv

# ファイルのimport
from predict_image import predict_image
from mutual_info_quantity import mutual_info_quantity
from global_value import EPOCHS, BATCH_SIZE, dataset_num, date, train_data_date



def main():

    calc_test_MI()
    calc_train_MI()


def calc_test_MI():
    # unused_fileの読み込み
    result_dirpath = date + "\\" + model_name + "\\"   # k_20
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = pickle.load(f)
    f.close()

    filename_all = ["ファイル名"]
    e_b_all = ["平均情報量"]  # originalの平均情報量のリスト
    MI_all = ["相互情報量"]  # 相互情報量のリスト

    for filename in unused_filename:
        # predict_image(filename)
        e_b, mutual = mutual_info_quantity(filename, "test")

        filename_all.append(filename)
        e_b_all.append(e_b)
        MI_all.append(mutual)

        print(f"{filename} is finished!!")

    # 情報量の保存
    MI_save_path = date + r"\predict\mutual_info"  # k_20
    MI_save_path = MI_save_path + f"\\mutual_info={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_test.csv"  # train

    with open(MI_save_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(filename_all)
        writer.writerow(e_b_all)
        writer.writerow(MI_all)


def calc_train_MI():

    result_dirpath = date + "\\" + model_name + "\\"
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = pickle.load(f)
    f.close()

    filename_all = ["ファイル名"]
    e_b_all = ["平均情報量"]  # originalの平均情報量のリスト
    MI_all = ["相互情報量"]  # 相互情報量のリスト

    for filename in used_filename:
        # predict_image(filename)
        e_b, mutual = mutual_info_quantity(filename, "train")

        filename_all.append(filename)
        e_b_all.append(e_b)
        MI_all.append(mutual)

        print(f"{filename} is finished!!")

    # 情報量の保存
    MI_save_path = date + r"\predict\mutual_info"  # k_20
    MI_save_path = MI_save_path + f"\\mutual_info={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_train.csv"

    with open(MI_save_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(filename_all)
        writer.writerow(e_b_all)
        writer.writerow(MI_all)


if __name__ == "__main__":
    main()
