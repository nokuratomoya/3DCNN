import os
import numpy as np
# import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
from global_value import E, BATCH_SIZE, dataset_num, date, time_size, xy_size, model_date, results_date, start_num, end_num
import pickle
from natsort import natsorted
import csv


def main():
    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"
    results_path = r"C:\Users\AIlab\labo\LNPmodel\results\\" + results_date + r"\predict\\"
    # eval_file_path = r"G:\LNImodel\train_data\20240704\gain2_dt0.05\0to99"
    eval_file_path = r"H:\G\LNPmodel\train_data\20240824\poisson_dt2e-05_dt2.5\0to99"

    # eval_file_all = natsorted(os.listdir(eval_file_path))
    # for eval_file in eval_file_all:
    calc_any_SSIM(model_path, results_path, eval_file_path)


# 学習したモデルの評価用
def calc_SSIM():
    result_folder = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\"
    model_name = f"result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}"

    # 訓練、テストデータファイル名の読み込み
    used_filename, unused_filename = read_file(result_folder + model_name)
    used_filename = natsorted(used_filename)
    unused_filename = natsorted(unused_filename)

    # SSIM計算
    # 訓練データSSIM
    train_SSIM = []
    for i in used_filename:
        ori_img, pre_img = image_open(i, result_folder + model_name, "train")
        train_SSIM.append(ssim(ori_img, pre_img))
        # print(f"finished:{i}")

    # テストデータSSIM
    test_SSIM = []
    for i in unused_filename:
        ori_img, pre_img = image_open(i, result_folder + model_name, "test")
        test_SSIM.append(ssim(ori_img, pre_img))
        # print(f"finished:{i}")

    # SSIM保存
    SSIM_save_path = result_folder + model_name + "\\predict\\SSIM"
    os.makedirs(SSIM_save_path, exist_ok=True)

    # csvに保存
    with open(SSIM_save_path + "\\SSIM.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(used_filename)
        writer.writerow(train_SSIM)
        writer.writerow(unused_filename)
        writer.writerow(test_SSIM)
    file.close()


def calc_any_SSIM(model_path, results_path, eval_file_path):
    eval_file_all = natsorted(os.listdir(eval_file_path))

    # 訓練、テストデータファイル名の読み込み
    model_result_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_path
    used_filename, unused_filename = read_file(model_result_path)
    used_filename = natsorted(used_filename)
    unused_filename = natsorted(unused_filename)

    # 評価画像が入ったフォルダのパス
    # result_folder = r"C:\Users\AIlab\labo\3DCNN\results\\" + results_date + "\\"
    # results_path

    train_SSIM = []
    test_SSIM = []
    total_SSIM = []
    for eval_file_name in eval_file_all:

        if eval_file_name in used_filename:
            ori_img, pre_img = image_open(eval_file_name, results_path, "train")
            train_SSIM.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))
            total_SSIM.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))
        elif eval_file_name in unused_filename:
            ori_img, pre_img = image_open(eval_file_name, results_path, "test")
            test_SSIM.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))
            total_SSIM.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))


    # SSIM保存
    SSIM_save_path = results_path + "SSIM"
    os.makedirs(SSIM_save_path, exist_ok=True)

    # csvに保存
    with open(SSIM_save_path + "\\SSIM.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(used_filename)
        writer.writerow(train_SSIM)
        writer.writerow(unused_filename)
        writer.writerow(test_SSIM)
    file.close()

    # total
    # csvに保存
    with open(SSIM_save_path + "\\SSIM_total.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eval_file_all)
        writer.writerow(total_SSIM)
    file.close()


def calc_3D_SSIM(model_path, results_path, eval_file_path):
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
        print(f"calc_SSIM:{eval_file_name}")
        SSIM_one = []

        if eval_file_name in used_filename:

            ori_img, pre_images = image_open_all(eval_file_name, results_path, "train", start_num, end_num)
            for pre_img in pre_images:
                SSIM_one.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))

        elif eval_file_name in unused_filename:
            ori_img, pre_images = image_open_all(eval_file_name, results_path, "test", start_num, end_num)
            for pre_img in pre_images:
                SSIM_one.append(ssim(ori_img, pre_img, data_range=pre_img.max() - pre_img.min()))

        SSIM_one_save_path = results_path + f"SSIM_3D\\"
        if time_size != 100:
            SSIM_one_save_path = results_path + f"SSIM_3D_{time_size}\\"
        os.makedirs(SSIM_one_save_path, exist_ok=True)

        # csvに保存
        with open(SSIM_one_save_path + f"{eval_file_name}_SSIM.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.arange(start_num, end_num + 1))
            writer.writerow(SSIM_one)

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


def image_open(filename, predict_path, t_v):
    original_dir = rf"H:\train_data\20240711\0to1199_resized\\" + filename + ".jpg"
    predict_dir = predict_path + f"{t_v}_predict" + "\\pre_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}.jpg"
    ori_img = Image.open(original_dir)
    pre_img = Image.open(predict_dir)
    ori_img = np.array(ori_img)
    pre_img = np.array(pre_img)

    return ori_img, pre_img


def image_open_all(filename, predict_path, t_v, start_num, end_num):
    original_dir = rf"H:\train_data\20240711\0to1199_resized\\" + filename + ".jpg"
    pre_images = []
    ori_img = Image.open(original_dir)
    for i in range(end_num - start_num + 1):
        preduct_dir = predict_path + f"{t_v}_predict_3D" + f"\\{filename}\\pre3D_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}_{i + start_num}.jpg"
        if time_size != 100:
            preduct_dir = predict_path + f"{t_v}_predict_3D_{time_size}" + f"\\{filename}\\pre3D_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}_{i + start_num}.jpg"
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
