import os
import numpy as np
# import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from PIL import Image
from global_value import E, BATCH_SIZE, dataset_num, date, time_size, xy_size
import pickle
from natsort import natsorted
import csv


def main():
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
    original_dir = predict_path + f"\\predict\\{t_v}_predict" + "\\pre_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}.jpg"
    predict_dir = predict_path + f"\\predict\\{t_v}_predict\\mutual_info\\" + filename + "_resize.jpg"
    ori_img = Image.open(predict_dir)
    pre_img = Image.open(original_dir)
    ori_img = np.array(ori_img)
    pre_img = np.array(pre_img)

    return ori_img, pre_img


if __name__ == "__main__":
    main()
