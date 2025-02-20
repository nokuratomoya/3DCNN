from global_value import model_date, time_size, xy_size, BATCH_SIZE, dataset_num, E, start_num, end_num
import os
from natsort import natsorted
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import math
import csv
import skimage.metrics



def main():
    spatial_filter_gain_loop = [1, 0.75, 0.5]
    temporal_filter_par_loop = [1, 2, 4, 8]
    loop_list = list(itertools.product(spatial_filter_gain_loop, temporal_filter_par_loop))
    a_list = [0.005375, 0.01075, 0.0215, 0.043, 0.086, 0.172, 0.344]
    filter_std_list = [3.8, 1.9, 7.6]
    # gain = spatial_filter_gain_loop[0]
    # par = temporal_filter_par_loop[0]

    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # a_compare
    for a in a_list:
        results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\20241217//a_compare//a={a}\\predict\\"
        eval_file_path = rf"E:\LNImodel\train_data\20241217//a={a}\gain24_dt2.5\0to99"
        # calc_3D_SSIM(model_path, results_predict_path, eval_file_path)
        calc_3D_MSE_or_PSNR(model_path, results_predict_path, eval_file_path, index="MSE")  # index : MSE or PSNR


def calc_3D_MSE_or_PSNR(model_path, results_path, eval_file_path, index="MSE"):
    eval_file_all = natsorted(os.listdir(eval_file_path))

    # 訓練、テストデータファイル名の読み込み
    model_result_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_path
    used_filename, unused_filename = read_file(model_result_path)
    used_filename = natsorted(used_filename)
    unused_filename = natsorted(unused_filename)

    # 評価画像が入ったフォルダのパス
    # result_folder = r"C:\Users\AIlab\labo\3DCNN\results\\" + results_date + "\\"
    # results_path

    for eval_file_name in eval_file_all:
        print(f"calc_{index}:{eval_file_name}")
        NCC_one = []

        if eval_file_name in used_filename:

            ori_img, pre_images = image_open_all(eval_file_name, results_path, "train", start_num, end_num)
            for pre_img in pre_images:
                # ori_img = np.ravel(ori_img)
                # pre_img = np.ravel(pre_img)
                if index == "PSNR":
                    NCC_one.append(skimage.metrics.peak_signal_noise_ratio(ori_img, pre_img, data_range=255))
                elif index == "MSE":
                    NCC_one.append(skimage.metrics.mean_squared_error(ori_img, pre_img))

        elif eval_file_name in unused_filename:
            ori_img, pre_images = image_open_all(eval_file_name, results_path, "test", start_num, end_num)
            for pre_img in pre_images:
                # ori_img = np.ravel(ori_img)
                # pre_img = np.ravel(pre_img)
                if index == "PSNR":
                    NCC_one.append(skimage.metrics.peak_signal_noise_ratio(ori_img, pre_img, data_range=255))
                elif index == "MSE":
                    NCC_one.append(skimage.metrics.mean_squared_error(ori_img, pre_img))

        SSIM_one_save_path = results_path + f"{index}_3D\\"
        os.makedirs(SSIM_one_save_path, exist_ok=True)

        # csvに保存
        with open(SSIM_one_save_path + f"{eval_file_name}_{index}.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.arange(start_num, end_num + 1))
            writer.writerow(NCC_one)

        file.close()


def read_file(folder_path):
    load_unused_dir = folder_path + f"\\model\\test_data={dataset_num}_e={E}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = pickle.load(f)
    f.close()

    load_used_dir = folder_path + f"\\model\\train_data={dataset_num}_e={E}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = pickle.load(f)
    f.close()

    return used_filename, unused_filename


def image_open_all(filename, predict_path, t_v, start_num, end_num):
    original_dir = rf"H:\train_data\20240711\0to1199_resized\\" + filename + ".jpg"
    pre_images = []
    ori_img = Image.open(original_dir)
    for i in range(end_num - start_num + 1):
        preduct_dir = predict_path + f"{t_v}_predict_3D" + f"\\{filename}\\pre3D_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}_{i + start_num}.jpg"
        pre_img_temp = Image.open(preduct_dir)
        pre_img_temp = np.array(pre_img_temp)
        pre_images.append(pre_img_temp)

    pre_img = np.array(pre_images)
    ori_img = np.array(ori_img)
    # print(pre_img.shape)
    # print(ori_img.shape)

    return ori_img, pre_img




if __name__ == "__main__":
    main()
