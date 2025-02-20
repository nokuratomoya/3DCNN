import dtcwt
import numpy as np
from global_value import model_date, time_size, xy_size, BATCH_SIZE, dataset_num, E, start_num, end_num
import os
from natsort import natsorted
import itertools
import matplotlib.pyplot as plt
from PIL import Image
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
        calc_3D_cw_SSIM(model_path, results_predict_path, eval_file_path)  # index : MSE or PSNR


def calc_3D_cw_SSIM(model_path, results_path, eval_file_path):
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
        print(f"calc_CW_SSIM:{eval_file_name}")
        NCC_one = []

        if eval_file_name in used_filename:

            ori_img, pre_images = image_open_all(eval_file_name, results_path, "train", start_num, end_num)
            for pre_img in pre_images:
                # ori_img = np.ravel(ori_img)
                # pre_img = np.ravel(pre_img)
                NCC_one.append(cw_ssim(ori_img, pre_img, nlevels=4))

        elif eval_file_name in unused_filename:
            ori_img, pre_images = image_open_all(eval_file_name, results_path, "test", start_num, end_num)
            for pre_img in pre_images:
                # ori_img = np.ravel(ori_img)
                # pre_img = np.ravel(pre_img)
                NCC_one.append(cw_ssim(ori_img, pre_img, nlevels=4))

        SSIM_one_save_path = results_path + f"CW_SSIM_3D\\"
        os.makedirs(SSIM_one_save_path, exist_ok=True)

        # csvに保存
        with open(SSIM_one_save_path + f"{eval_file_name}_CW_SSIM.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.arange(start_num, end_num + 1))
            writer.writerow(NCC_one)

        file.close()


def cw_ssim(image1: np.ndarray, image2: np.ndarray, nlevels=4, K=1e-6):
    """
    Complex Wavelet SSIM を二木方向複素ウェーブレット変換 (DTCWT) を用いて計算する簡易サンプル関数。

    Parameters
    ----------
    image1 : np.ndarray
        比較対象の画像1 (グレースケール想定)
    image2 : np.ndarray
        比較対象の画像2 (グレースケール想定)
    nlevels : int
        ウェーブレット分解の階層数
    K : float
        数値安定化のための小さな定数

    Returns
    -------
    cwssim_value : float
        CW-SSIM のスコア (1.0 に近いほど類似)
    """
    # 入力画像が同じサイズ・同じ型であることを前提
    # 必要に応じてリサイズ/型変換する
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # DTCWT 変換器を初期化
    transform = dtcwt.Transform2d()

    # 画像を複素ウェーブレット変換
    # lo: 低周波サブバンド, hi: 高周波サブバンド(複素数)
    coeffs1 = transform.forward(image1, nlevels=nlevels)
    coeffs2 = transform.forward(image2, nlevels=nlevels)

    # スケール・方向毎の CW-SSIM を計算し、平均を取る
    # lo 通過は実数なので、ここでは高周波部分(hi)のみを使うことが多い
    cwssim_scales = []

    for level in range(nlevels):
        # coeffs1.hi[level], coeffs2.hi[level] は shape=(方向数, H, W) の複素数配列
        subband1 = coeffs1.highpasses[level]  # (6, H, W) の複素数
        subband2 = coeffs2.highpasses[level]  # (6, H, W) の複素数

        n_orient, _, _ = subband1.shape
        cwssim_orients = []

        for orient in range(n_orient):
            c1 = subband1[orient]
            c2 = subband2[orient]

            # CW-SSIM の簡易的なグローバル版を計算 (局所ウィンドウではなく画像全域)
            # 参考式: CW-SSIM = (2 * Σ|X_i Y_i*| + K) / (Σ|X_i|^2 + Σ|Y_i|^2 + K)
            numerator = 2.0 * np.sum(np.abs(c1 * np.conjugate(c2))) + K
            denominator = np.sum(np.abs(c1) ** 2) + np.sum(np.abs(c2) ** 2) + K
            cw_ssim_val = np.real(numerator / denominator)  # 実数部分をとる

            cwssim_orients.append(cw_ssim_val)

        # このスケールでの平均を取る
        cwssim_scales.append(np.mean(cwssim_orients))

    # 全スケールの平均を最終スコアとする (手法によっては重み付けあり)
    cwssim_value = np.mean(cwssim_scales)
    return cwssim_value


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
