# predict_imageをxy_sizeを変えて複数回す用
# EPOCHSだけ自分で設定する必要あり

from tensorflow.keras.models import load_model
from func_3DCNN_predict import load_dataset_predict, img_show, pre_concatenate, pre_npy_save, load_dataset_predict_3D
import numpy as np
import os
import pickle
from global_value import BATCH_SIZE, dataset_num, model_date, pixel_size, split_num, model_dim, pre_start_num, \
    pre_end_num, output3D, time_size
from natsort import natsorted
import tensorflow as tf
import time

global predict_time_total

predict_file_path = ""


def predict_calc_save(pre_file, EPOCHS, model, save_path):
    global predict_time_total
    # pre_file = '221017_data'
    x_pre = load_dataset_predict(pre_file, predict_file_path)
    x_pre = np.array(x_pre)
    print("2D x_pre.shape:", x_pre.shape)
    # print(x_pre.shape)
    if model_dim == "SID":
        x_pre = x_pre.reshape(x_pre.shape[0], x_pre.shape[2], x_pre.shape[3])
        x_pre = x_pre[:, :, :, np.newaxis]
    else:
        x_pre = x_pre[:, :, :, :, np.newaxis]

    print("x_pre.shape:", x_pre.shape)

    start_time = time.perf_counter()
    x_pre = model.predict(x_pre)
    end_time = time.perf_counter()
    print(f"predict_time:{end_time - start_time}s")
    predict_time_total += end_time - start_time
    # print(x_pre.shape)
    x_pre *= 255.0
    x_pre = x_pre.reshape(split_num * split_num, pixel_size, pixel_size)

    # concatしなくてもうまくいく？
    x_pre = x_pre.reshape(pixel_size, pixel_size)
    x_pre_total = x_pre

    filename = f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'
    # pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'  # e=EPOCHS, # b=BATCH_SIZE

    pre_filepath = save_path + filename
    os.makedirs(save_path + "save_npy", exist_ok=True)

    # 画像保存
    img_show(x_pre_total, pre_filepath)

    # npy保存
    pre_npy_save(x_pre_total, save_path + f'save_npy\\' + filename)


def predict_3D_calc_save(pre_file, EPOCHS, model, save_path):
    print(save_path)

    global predict_time_total
    # pre_file = '221017_data'
    x_pres = load_dataset_predict_3D(pre_file, pre_start_num, pre_end_num, predict_file_path)
    x_pres = np.array(x_pres)
    # print(x_pres.shape)

    x_pres = x_pres[:, :, :, :, np.newaxis]
    # print("x_pre.shape", x_pres.shape)  # (1, 50, 120, 120, 1)

    start_time = time.perf_counter()
    # for i, x_pre in enumerate(x_pres):
    x_pres_results = model.predict(x_pres)

    x_pres_results *= 255.0
    x_pres_results = x_pres_results.reshape(pre_end_num - pre_start_num + 1, pixel_size, pixel_size)
    x_pres_results = np.array(x_pres_results)
    # x_pres_results = model.predict(x_pres)

    # print("x_pres_results.shape", x_pres_results.shape)
    # x_pre = model.predict(x_pres)

    end_time = time.perf_counter()
    print(f"predict_time:{end_time - start_time}s")
    predict_time_total += end_time - start_time
    # print(x_pre.shape)
    # x_pres_results *= 255.0
    # x_pres_results = x_pres_results.reshape(pre_end_num - pre_start_num, pixel_size, pixel_size)

    # save_path = save_path + f'{pre_file}\\'

    # pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'  # e=EPOCHS, # b=BATCH_SIZE

    os.makedirs(save_path + f"{pre_file}", exist_ok=True)
    os.makedirs(save_path + f"save_npy\\{pre_file}", exist_ok=True)

    for i, x_pre_results in enumerate(x_pres_results):
        filename = f'pre3D_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i + pre_start_num}'
        pre_filepath = save_path + f"{pre_file}\\" + filename
        # 画像保存
        img_show(x_pre_results, pre_filepath)

        # npy保存
        pre_npy_save(x_pre_results, save_path + f'save_npy\\{pre_file}\\' + filename)


def predict_image(model_name, EPOCHS):
    global predict_time_total
    predict_time_total = 0
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + r'\\' + model_name + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    # model = load_model(model_h5)
    model = load_model(model_h5, custom_objects={"ssim_loss": ssim_loss})

    # unused_data(テストデータ)
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = natsorted(pickle.load(f))
    f.close()
    save_path = result_dirpath + r'\predict\\test_predict\\'
    os.makedirs(save_path + "save_npy", exist_ok=True)
    for i in unused_filename:
        predict_calc_save(i, EPOCHS, model, save_path)
        print(f"finished:{i}")

    # used(訓練データ)
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = natsorted(pickle.load(f))
    f.close()
    save_path = result_dirpath + r'\predict\\train_predict\\'
    os.makedirs(save_path + "save_npy", exist_ok=True)
    for i in used_filename:
        predict_calc_save(i, EPOCHS, model, save_path)
        print(f"finished:{i}")

    print(f"predict_time_total:{predict_time_total}s")
    print(f"predict_time_average:{predict_time_total / (len(unused_filename) + len(used_filename))}s")


def predict_any_image(model_path, EPOCHS, pre_file_name, save_path, predict_file_path_t):
    global predict_time_total
    global predict_file_path

    print("save_path", save_path)

    predict_file_path = predict_file_path_t
    predict_time_total = 0
    # 学習済みmodelのロード
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_path + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    # model = load_model(model_h5)
    model = load_model(model_h5, custom_objects={"ssim_loss": ssim_loss})
    # print("model_input_shape", model.input_shape)

    # 訓練データ、テストデータのファイル名を取得
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = natsorted(pickle.load(f))
    f.close()
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = natsorted(pickle.load(f))
    f.close()

    # 画像の予測、保存
    if pre_file_name in unused_filename:
        if output3D:
            save_path = save_path + f"test_predict_3D_{time_size}\\"
            predict_3D_calc_save(pre_file_name, EPOCHS, model, save_path)
        else:
            save_path = save_path + f"test_predict_{time_size}\\"  # 保存先
            predict_calc_save(pre_file_name, EPOCHS, model, save_path)

    elif pre_file_name in used_filename:
        if output3D:
            save_path = save_path + f"train_predict_3D_{time_size}\\"
            predict_3D_calc_save(pre_file_name, EPOCHS, model, save_path)
        else:
            save_path = save_path + f"train_predict_{time_size}\\"  # 保存先
            predict_calc_save(pre_file_name, EPOCHS, model, save_path)

    # テストとして復元
    else:
        # print("ファイル名が存在しません")
        if output3D:
            save_path = save_path + f"test_predict_3D_{time_size}\\"
            predict_3D_calc_save(pre_file_name, EPOCHS, model, save_path)
        else:
            save_path = save_path + f"test_predict_{time_size}\\"  # 保存先
            predict_calc_save(pre_file_name, EPOCHS, model, save_path)


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,
                                            max_val=1.0, filter_size=11,
                                            filter_sigma=1.5, k1=0.01, k2=0.03))
