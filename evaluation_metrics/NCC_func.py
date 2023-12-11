import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import math
import os
import csv
from global_value import BATCH_SIZE, dataset_num, date
from natsort import natsorted
from scipy.stats import spearmanr


def NCC(model_name, EPOCHS):
    abs_path = r"C:\Users\AIlab\labo\3DCNN\results\\"
    # 訓練データで用いた画像
    # train_pre_data_dir = abs_path + date + r"\predict_k100\train_predict"
    # train_resize_dir = abs_path + date + r"\predict_k100\train_predict\mutual_info"
    train_pre_data_dir = abs_path + date + '\\' + model_name + r"\predict\train_predict"
    train_resize_dir = abs_path + date + '\\' + model_name + r"\predict\train_predict\mutual_info"

    # 評価用データ
    # vali_pre_data_dir = abs_path + date + r"\predict_k100\test_predict"
    # vali_resize_dir = abs_path + date + r"\predict_k100\test_predict\mutual_info"
    vali_pre_data_dir = abs_path + date + '\\' + model_name + r"\predict\test_predict"
    vali_resize_dir = abs_path + date + '\\' + model_name + r"\predict\test_predict\mutual_info"
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
        original_dir = train_pre_data_dir + "\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.jpg"

        predict_dir = train_resize_dir + "\\" + i + "_resize.jpg"

        ####
        ori_img = Image.open(predict_dir)
        pre_img = Image.open(original_dir)
        # numpy配列に
        ori_data = np.array(ori_img)
        pre_data = np.array(pre_img)
        # 平坦化
        ori_data = np.ravel(ori_data)
        pre_data = np.ravel(pre_data)

        NCC_plot_train(NCC_save_dir, i, ori_data, pre_data)
        """
        # 内積 (http://www.sanko-shoko.net/note.php?id=f7j3)
        inn_1 = int(np.correlate(ori_data, pre_data))  # a, b
        inn_2 = int(np.correlate(ori_data, ori_data))  # a, a
        inn_3 = int(np.correlate(pre_data, pre_data))  # b, b
        NCC_num = inn_1 / math.sqrt((inn_2 * inn_3))
        """

        NCC_train.append(NCC_calc(ori_data, pre_data))
        spearman_train.append(spearmanr(ori_data, pre_data)[0])

    # NCC_vali = ["評価データ"]
    NCC_vali = []
    # spearman_vali = ["評価データ"]
    spearman_vali = []
    # 評価データ
    for i in unused_filename:
        original_dir = vali_pre_data_dir + "\\pre_" + i + f"_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.jpg"
        predict_dir = vali_resize_dir + "\\" + i + "_resize.jpg"
        ori_img = Image.open(predict_dir)
        pre_img = Image.open(original_dir)
        # numpy配列に
        ori_data = np.array(ori_img)
        pre_data = np.array(pre_img)
        # 平坦化
        ori_data = np.ravel(ori_data)
        pre_data = np.ravel(pre_data)

        NCC_plot_test(NCC_save_dir, i, ori_data, pre_data)
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
        NCC_vali.append(NCC_calc(ori_data, pre_data))
        # スピアマンの順位相関係数を求める
        spearman_vali.append(spearmanr(ori_data, pre_data)[0])

    NCC_save_path = NCC_save_dir + "\\NCC_total.csv"
    spearman_save_path = NCC_save_dir + "\\spearman_total.csv"

    # used_filename.insert(0, "")
    # unused_filename.insert(0, "")
    # 値の整理
    # csvに保存
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


