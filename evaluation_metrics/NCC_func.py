import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import math
import os
import csv
from global_value import BATCH_SIZE, dataset_num, date, E, start_num, end_num
from natsort import natsorted
from scipy.stats import spearmanr

plot_bool = False
NCC_bool = False


def NCC(model_name, EPOCHS):
    abs_path = r"C:\Users\AIlab\labo\3DCNN\results\\"
    # 訓練データで用いた画像
    # train_pre_data_dir = abs_path + date + r"\predict_k100\train_predict"
    # train_resize_dir = abs_path + date + r"\predict_k100\train_predict\mutual_info"
    train_pre_data_dir = abs_path + date + '\\' + model_name + r"\predict\train_predict"
    train_resize_dir = abs_path + date + '\\' + model_name + r"\predict\train_predict\resize_img"

    # 評価用データ
    # vali_pre_data_dir = abs_path + date + r"\predict_k100\test_predict"
    # vali_resize_dir = abs_path + date + r"\predict_k100\test_predict\mutual_info"
    vali_pre_data_dir = abs_path + date + '\\' + model_name + r"\predict\test_predict"
    vali_resize_dir = abs_path + date + '\\' + model_name + r"\predict\test_predict\resize_img"
    # 60枚すべてのNCC求める
    ########################

    result_dirpath = abs_path + date + '\\' + model_name + '\\'  # k_20
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = natsorted(pickle.load(f))
    f.close()

    # dirname_main = r'C:\Users\AIlab\labo\複合化用NNプログラム_3Dver\train_data\20230120'
    # all_filename = os.listdir(dirname_main)
    # used_filename = set(all_filename) ^ set(unused_filename)
    # used_filename = list(used_filename)

    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = natsorted(pickle.load(f))
    f.close()
    #########################
    # NCCセーブフォルダの作成
    NCC_save_dir = f"C:\\Users\\AIlab\\labo\\3DCNN\\results\\" + date + "\\" + model_name + f"\\predict\\NCC\\dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}"
    os.makedirs(NCC_save_dir, exist_ok=True)
    os.makedirs(NCC_save_dir + "\\train", exist_ok=True)
    os.makedirs(NCC_save_dir + "\\test", exist_ok=True)

    # NCC_train = ["訓練データ"]
    NCC_train = []
    # spearman_train = ["訓練データ"]
    spearman_train = []
    # 訓練データ
    for i in used_filename:
        # original_dir = train_pre_data_dir + "\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.jpg"
        #
        # predict_dir = train_resize_dir + "\\" + i + "_resize.jpg"
        #
        # ####
        # ori_img = Image.open(predict_dir)
        # pre_img = Image.open(original_dir)
        # # numpy配列に
        # ori_data = np.array(ori_img)
        # pre_data = np.array(pre_img)
        #####################
        original_dir = train_pre_data_dir + "\\save_npy\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.npy"

        predict_dir = train_resize_dir + "\\save_npy\\" + i + "_resize.npy"

        ori_data = np.load(original_dir)
        pre_data = np.load(predict_dir)
        ###########################

        # 平坦化
        ori_data = np.ravel(ori_data)
        pre_data = np.ravel(pre_data)
        if plot_bool:
            NCC_plot_train(NCC_save_dir, i, ori_data, pre_data)
        # NCC_plot_train(NCC_save_dir, i, ori_data, pre_data)
        """
        # 内積 (http://www.sanko-shoko.net/note.php?id=f7j3)
        inn_1 = int(np.correlate(ori_data, pre_data))  # a, b
        inn_2 = int(np.correlate(ori_data, ori_data))  # a, a
        inn_3 = int(np.correlate(pre_data, pre_data))  # b, b
        NCC_num = inn_1 / math.sqrt((inn_2 * inn_3))
        """
        if NCC_bool:
            NCC_train.append(NCC_calc(ori_data, pre_data))
        spearman_train.append(spearmanr(ori_data, pre_data)[0])

        print(f"{i} is finished!!")

    # NCC_vali = ["評価データ"]
    NCC_vali = []
    # spearman_vali = ["評価データ"]
    spearman_vali = []
    # 評価データ
    for i in unused_filename:
        # original_dir = vali_pre_data_dir + "\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.jpg"
        # predict_dir = vali_resize_dir + "\\" + i + "_resize.jpg"
        # ori_img = Image.open(predict_dir)
        # pre_img = Image.open(original_dir)
        # # numpy配列に
        # ori_data = np.array(ori_img)
        # pre_data = np.array(pre_img)

        original_dir = vali_pre_data_dir + "\\save_npy\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.npy"

        predict_dir = vali_resize_dir + "\\save_npy\\" + i + "_resize.npy"

        ori_data = np.load(original_dir)
        pre_data = np.load(predict_dir)


        # 平坦化
        ori_data = np.ravel(ori_data)
        pre_data = np.ravel(pre_data)
        if plot_bool:
            NCC_plot_test(NCC_save_dir, i, ori_data, pre_data)
        # NCC_plot_test(NCC_save_dir, i, ori_data, pre_data)
        """
        # 内積 (http://www.sanko-shoko.net/note.php?id=f7j3)
        inn_1 = int(np.correlate(ori_data, pre_data))  # a, b
        inn_2 = int(np.correlate(ori_data, ori_data))  # a, a
        inn_3 = int(np.correlate(pre_data, pre_data))  # b, b
        NCC_num = inn_1 / math.sqrt((inn_2 * inn_3))
        """
        # e1, e2, e3 = 0, 0, 0
        # for j in range(len(ori_data)):
        #     t1 = (ori_data[j] - np.mean(ori_data))
        #     t2 = (pre_data[j] - np.mean(pre_data))
        #     e2 += t1 * t1
        #     e3 += t2 * t2
        #     e1 += t1 * t2
        #
        # NCC_vali_num = e1 / (math.sqrt(e2) * math.sqrt(e3))
        # 相関係数を求める
        if NCC_bool:
            NCC_vali.append(NCC_calc(ori_data, pre_data))
        # スピアマンの順位相関係数を求める
        spearman_vali.append(spearmanr(ori_data, pre_data)[0])

        print(f"{i} is finished!!")

    NCC_save_path = NCC_save_dir + "\\NCC_total.csv"
    spearman_save_path = NCC_save_dir + "\\spearman_total.csv"

    # used_filename.insert(0, "")
    # unused_filename.insert(0, "")
    # 値の整理
    # csvに保存
    if NCC_bool:
        with open(NCC_save_path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(used_filename)
            writer.writerow(NCC_train)
            writer.writerow(unused_filename)
            writer.writerow(NCC_vali)

        f.close()

    with open(spearman_save_path, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(used_filename)
        writer.writerow(spearman_train)
        writer.writerow(unused_filename)
        writer.writerow(spearman_vali)

    f.close()


def calc_any_NCC(model_path, results_path, eval_file_path):
    eval_file_all = natsorted(os.listdir(eval_file_path))

    # 訓練、テストデータファイル名の読み込み
    model_result_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_path
    used_filename, unused_filename = read_file(model_result_path)
    used_filename = natsorted(used_filename)
    unused_filename = natsorted(unused_filename)

    train_NCC_spear = []
    test_NCC_spear = []
    total_NCC_spear = []
    for eval_file_name in eval_file_all:
        if eval_file_name in used_filename:
            ori_data, pre_data = image_open_npy(eval_file_name, results_path, "train")
            ori_data = np.ravel(ori_data)
            pre_data = np.ravel(pre_data)
            train_NCC_spear.append(spearmanr(ori_data, pre_data)[0])
            total_NCC_spear.append(spearmanr(ori_data, pre_data)[0])

            # train_NCC.append(ssim(ori_data, pre_data))
        elif eval_file_name in unused_filename:
            ori_data, pre_data = image_open_npy(eval_file_name, results_path, "test")
            ori_data = np.ravel(ori_data)
            pre_data = np.ravel(pre_data)
            test_NCC_spear.append(spearmanr(ori_data, pre_data)[0])
            total_NCC_spear.append(spearmanr(ori_data, pre_data)[0])

    # spearman保存
    NCC_spear_save_path = results_path + "NCC_spearman"
    os.makedirs(NCC_spear_save_path, exist_ok=True)

    # csvに保存
    with open(NCC_spear_save_path + "\\spearman_NCC.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(used_filename)
        writer.writerow(train_NCC_spear)
        writer.writerow(unused_filename)
        writer.writerow(test_NCC_spear)
    file.close()

    with open(NCC_spear_save_path + "\\spearman_NCC_total.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eval_file_all)
        writer.writerow(total_NCC_spear)

    file.close()


def calc_3D_NCC(model_path, results_path, eval_file_path):
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
        print(f"calc_NCCspearman:{eval_file_name}")
        NCC_one = []

        if eval_file_name in used_filename:

            ori_img, pre_images = image_open_all(eval_file_name, results_path, "train", start_num, end_num)
            for pre_img in pre_images:
                ori_img = np.ravel(ori_img)
                pre_img = np.ravel(pre_img)
                NCC_one.append(spearmanr(ori_img, pre_img)[0])

        elif eval_file_name in unused_filename:
            ori_img, pre_images = image_open_all(eval_file_name, results_path, "test", start_num, end_num)
            for pre_img in pre_images:
                ori_img = np.ravel(ori_img)
                pre_img = np.ravel(pre_img)
                NCC_one.append(spearmanr(ori_img, pre_img)[0])

        SSIM_one_save_path = results_path + "NCCspearman_3D\\"
        os.makedirs(SSIM_one_save_path, exist_ok=True)

        # csvに保存
        with open(SSIM_one_save_path + f"{eval_file_name}_NCCspearman.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(np.arange(start_num, end_num + 1))
            writer.writerow(NCC_one)

        file.close()


def NCC_calc(ori_data, pre_data):
    # 内積 (http://www.sanko-shoko.net/note.php?id=f7j3)
    e1, e2, e3 = 0, 0, 0
    for j in range(len(ori_data)):
        t1 = (ori_data[j] - np.mean(ori_data))
        t2 = (pre_data[j] - np.mean(pre_data))
        e2 += t1 * t1
        e3 += t2 * t2
        e1 += t1 * t2
    NCC_vali_num = e1 / (math.sqrt(e2) * math.sqrt(e3))
    return NCC_vali_num

#
# def spearmanr_calc(ori_data, pre_data):
#     correlation, pvalue = spearmanr(ori_data, pre_data)
#     return correlation


def NCC_plot_train(NCC_save_dir, file_name, ori_img_1dim, pre_img_1dim):

    # plt.figure(figsize=(8, 8), tight_layout=True)
    plt.figure(figsize=(8, 8))
    plt.rcParams["font.size"] = 30
    # plt.figure()
    plt.xlim(0, 255)
    plt.xticks(np.arange(0, 255, 250))
    plt.ylim(0, 255)
    plt.yticks(np.arange(0, 255, 250))
    # plt.rcParams["font.size"] = 20
    # plt.axis("off")
    #  plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.scatter(ori_img_1dim, pre_img_1dim, s=1, c="tab:blue")
    plt.savefig(NCC_save_dir + f"\\train\\{file_name}_cc.jpg")
    plt.close()
    # plt.show()
    print(f"{file_name} is finished!!")


def NCC_plot_test(NCC_save_dir, file_name, ori_img_1dim, pre_img_1dim):
    # plt.figure(figsize=(8, 8), tight_layout=True)
    plt.figure(figsize=(8, 8))
    # plt.figure()
    plt.xlim(0, 255)
    plt.xticks(np.arange(0, 255, 250))
    plt.ylim(0, 255)
    plt.yticks(np.arange(0, 255, 250))
    # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.scatter(ori_img_1dim, pre_img_1dim, s=1, c="tab:green")

    plt.savefig(NCC_save_dir + f"\\test\\{file_name}_cc.jpg")
    # plt.show()
    plt.close()
    print(f"{file_name} is finished!!")


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


def image_open_npy(filename, predict_path, t_v):
    original_dir = predict_path + f"{t_v}_predict" + "\\save_npy\\pre_" + filename + f"_dataset={dataset_num}_e={E}_b={BATCH_SIZE}.npy"
    predict_dir = rf"H:\train_data\20240625\0to1199_resized\\save_npy\\" + filename + ".npy"

    ori_img_npy = np.load(original_dir)
    pre_img_npy = np.load(predict_dir)

    return ori_img_npy, pre_img_npy


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


