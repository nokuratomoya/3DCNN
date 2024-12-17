from predict_img_func import predict_image, predict_any_image
from global_value import time_size, xy_size, E, model_date, get_now, gain_total, par_total, predict_file_path
import os
from natsort import natsorted
import csv
import numpy as np
from tensorflow.keras.models import load_model
import itertools
import tensorflow as tf
from func_3DCNN_predict import img_show, pre_npy_save
from PIL import Image



# E = 1794
# time_size = 100
# xy_size = 30

# predict_file_path = ""
# save_path = ""

def main():
    BATCH_SIZE = 16
    dataset_total = 1200
    dataset_num = int(dataset_total * 0.8)
    model_date = "20240810"
    time_size = 100
    xy_size = 3
    pixel_size = 120
    split_num = 1  # 分割サイズ　split_num*split_num分割される
    EPOCHS = 1386
    pre_start_num = 100
    pre_end_num = 800

    filename = "retina"
    # save_path = rf"C:\Users\AIlab\labo\onishi\results_resize\{filename}\predict\\"
    # npz_path = rf"C:\Users\AIlab\labo\onishi\results_expanded\results_expanded_{filename}.npz"
    # npz = np.load(npz_path)
    pre_file = "retina"  # TCR, IN, TRN
    # spike_data = npz[pre_file]
    retina_spike_path = r"H:\train_data\onishi\onishi_resize120\img1\\"
    retina_spike_file_all = natsorted(os.listdir(retina_spike_path))

    load_npy_path_all = list(
        map(lambda x: retina_spike_path + str(x), retina_spike_file_all))

    spike_data = list(map(lambda x: np.load(x), load_npy_path_all))
    spike_data = np.array(spike_data)
    save_path = rf"C:\Users\AIlab\labo\onishi\results_resize\{filename}\predict\\"


    os.makedirs(save_path + f"{pre_file}", exist_ok=True)
    os.makedirs(save_path + f"save_npy\\{pre_file}", exist_ok=True)
    # スパイクの編集
    # 圧縮（0.5ms -> 5ms）
    # spike_data = compressed_spike(spike_data)

    # 画像サイズの変更(64*64 -> 120*120)
    # spike_data = padding_img(spike_data)



    # 128*128 -> 120*120
    time, x, y = spike_data.shape
    start = int((x - pixel_size) / 2)
    end = int((x + pixel_size) / 2)
    spike_data = spike_data[:, start:end, start:end]

    print("spike_data.shape:", spike_data.shape)

    # 配列の入れ替え([802, 120, 120] -> [(復元する画像の枚数), time_size, 120, 120])

    # 新しい配列の形を指定 (dim=701, 100, 120, 120)
    new_array = np.zeros((pre_end_num - pre_start_num + 1, time_size, pixel_size, pixel_size))
    # new_array = []
    for i in range(pre_end_num - pre_start_num + 1):
        new_array[i] = spike_data[pre_start_num - time_size + i:i + pre_start_num]

    x_pres = new_array[:, :, :, :, np.newaxis]

    os.makedirs(save_path, exist_ok=True)
    # model_dateの保存
    save_data_csv(["model_date",
                   "time_size",
                   "xy_size",
                   "EPOCHS",
                   "predict_file_path",
                   ], save_path + "trained_model.csv")

    save_data_csv([model_date,
                   time_size,
                   xy_size,
                   E,
                   predict_file_path,
                   ], save_path + "trained_model.csv")

    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # 保存先のディレクトリ

    # 学習済みmodelのロード
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_path + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    # model = load_model(model_h5)
    model = load_model(model_h5, custom_objects={"ssim_loss": ssim_loss})

    # 推論
    x_pres_results = model.predict(x_pres)
    print("x_pres_results.shape:", x_pres_results.shape)
    x_pres_results *= 255.0
    x_pres_results = x_pres_results.reshape(pre_end_num - pre_start_num + 1, pixel_size, pixel_size)
    x_pres_results = np.array(x_pres_results)

    for i, x_pre_results in enumerate(x_pres_results):
        filename = f'pre3D_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i + pre_start_num}'
        pre_filepath = save_path + f"{pre_file}\\" + filename
        # 画像保存
        img_show(x_pre_results, pre_filepath)

        # npy保存
        pre_npy_save(x_pre_results, save_path + f'save_npy\\{pre_file}\\' + filename)


def retina_spike_input():
    retina_spike_path = r"H:\train_data\20240711\0to1199\img0\img1\\"
    retina_spike_file_all = natsorted(os.listdir(retina_spike_path))

    load_npy_path_all = list(
        map(lambda x: retina_spike_path + str(x), retina_spike_file_all))

    retina_spike_all = list(map(lambda x: np.load(x), load_npy_path_all))
    # retina_spike_all : [800, 128, 128]
    retina_spike_all = np.array(retina_spike_all)
    retina_spike_all_120 = retina_spike_all[:, 4:124, 4:124]
    # retina_spike_all : [800, 120, 120]

    # 画像として保存する関数
    def save_images(images, output_dir):
        for i, image in enumerate(images):
            ### npy保存
            np.save(os.path.join(output_dir, "img1", f"img1_{i}.npy"), image)

            ### image保存
            # 二値データを0-255の範囲に変換
            image_to_save = ((1 - image) * 255).astype(np.uint8)


            # PILのImageオブジェクトに変換
            img = Image.fromarray(image_to_save)

            # ファイル名を作成して保存
            file_path = os.path.join(output_dir, "image\\img1", f"img1_{i}.png")  # 例: image_0001.png
            img.save(file_path)

    # [120, 120]の画像を[60, 60]にリサイズする関数
    ###################################################
    def downsample_to_binary_2d(image):
        # 128x128の画像を2x2のブロックごとに分割
        h, w = image.shape
        reshaped = image.reshape(h // 2, 2, w // 2, 2)

        # 各2x2ブロックに「1が一つでもあるか」を判定
        downsampled = reshaped.max(axis=(1, 3))

        return downsampled

    # [800, 128, 128]の画像を[800, 64, 64]にリサイズする関数
    def downsample_to_binary_3d(images):
        # 各スライス（800枚）に対して処理を適用
        return np.array([downsample_to_binary_2d(image) for image in images])

    # リサイズ処理
    retina_spike_all_60 = downsample_to_binary_3d(retina_spike_all_120)

    # 保存先のディレクトリを指定
    output_dir = r"H:\train_data\onishi_resize60"
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成
    os.makedirs(output_dir + "\\image\\img1", exist_ok=True)  # ディレクトリがなければ作成

    # 画像を保存
    save_images(retina_spike_all_60, output_dir)

    ###################################################

    # 60x60の画像を120x120にリサイズする関数
    def upsample_to_binary_2d(image):
        # 各ピクセルを2x2に複製
        upsampled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
        return upsampled

    # [800, 60, 60]を[800, 120, 120]にリサイズする関数
    def upsample_to_binary_3d(images):
        # 各スライス（800枚）に対して処理を適用
        return np.array([upsample_to_binary_2d(image) for image in images])

    # リサイズ処理
    retina_spike_all_120_resize = upsample_to_binary_3d(retina_spike_all_60)

    ############################################################
    # 保存先のディレクトリを指定
    output_dir = r"H:\train_data\onishi_resize120"
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成

    # 画像を保存
    save_images(retina_spike_all_120_resize, output_dir)




def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)

        # データを追加して保存する
        writer.writerow(save_data)

    file.close()


def compressed_spike(spike_data):
    compressed_times = 10
    compressed_array = np.zeros((int(len(spike_data) / compressed_times), spike_data.shape[1], spike_data.shape[2]))
    # skip_count = int(original_dt / dt)

    for l in range(spike_data.shape[1]):
        for m in range(spike_data.shape[2]):
            for n in range(int(len(spike_data) / compressed_times)):
                start_index = n * compressed_times
                end_index = start_index + compressed_times
                if np.any(spike_data[start_index:end_index, l, m] == 1):
                    compressed_array[n, l, m] = 1
    return compressed_array


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,
                                            max_val=1.0, filter_size=11,
                                            filter_sigma=1.5, k1=0.01, k2=0.03))


# 64*64の画像を120*120になるように周辺を０で埋めるプログラムを書いて
def padding_img(img):
    t, x, y = img.shape
    img_size = 120
    start = (img_size - x) // 2
    end = (img_size + x) // 2
    img_pad = np.zeros((t, img_size, img_size))
    img_pad[:, start:end, start:end] = img
    return img_pad


if __name__ == "__main__":
    main()
