from global_value import time_size, xy_size, E
from tensorflow.keras.models import load_model
from func_3DCNN_predict import load_dataset_predict_3D, img_show, pre_concatenate, pre_npy_save, load_spike_data
import numpy as np
import os
import pickle
from global_value import BATCH_SIZE, dataset_num, model_date, split_num, pixel_size
from natsort import natsorted
import tensorflow as tf

def main():
    model_name = f"result_kernel_{time_size}_{xy_size}_{xy_size}"
    # predict_image_3D(model_name, E)
    predict_movie_3D(model_name, E)


def predict_movie_3D(model_name, EPOCHS):
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + r'\\' + model_name + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    # model = load_model(model_h5)
    model = load_model(model_h5, custom_objects={"ssim_loss": ssim_loss})

    spike_dir_name = r"F:\train_data\20240212\movie\img0\\"
    # spike_dir_name = r"F:\train_data\20240108\0to1199\img0\\"

    spike_file_name = spike_dir_name + "img1"
    x_pre = load_spike_data(spike_file_name)
    x_pre = np.array(x_pre)
    x_pre = x_pre[:, :, :, :, np.newaxis]

    # x_pre = x_pre[0:3000, :, :, :, :]

    x_pre = model.predict(x_pre)
    x_pre *= 255.0

    x_pre_total = x_pre.reshape(len(x_pre), pixel_size, pixel_size)

    # pre_dirpath = result_dirpath + r'\predict\\' + t_v + f'_predict_3D\\{pre_file}\\'
    # pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i}'  # e=EPOCHS, # b=BATCH_SIZE
    # os.makedirs(pre_dirpath + "save_npy", exist_ok=True)
    # img_show(x_pre_total, pre_filepath)
    # pre_npy_save(x_pre_total,
    #              pre_dirpath + f'save_npy\\pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i}')

    for i in range(len(x_pre_total)):
        save_dir = r"C:\Users\AIlab\labo\3DCNN\movie_results\\" + model_date + r'\\' + "img0_1\\"
        os.makedirs(save_dir, exist_ok=True)
        save_file = save_dir + f'pre_{i}'
        img_show(x_pre_total[i], save_file)


def predict_image_3D(model_name, EPOCHS):

    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + r'\\' + model_name + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    # model = load_model(model_h5)
    model = load_model(model_h5, custom_objects={"ssim_loss": ssim_loss})

    # used(訓練データ)
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = natsorted(pickle.load(f))

    # 10個だけにする
    used_filename = used_filename[:3]
    f.close()
    for i in used_filename:
        predict_calc_3D(i, "train", result_dirpath, EPOCHS, model)
        print(f"finished:{i}")


def predict_calc_3D(pre_file, t_v, result_dirpath, EPOCHS, model):
    # pre_file = '221017_data'
    for i in range(700):
        # x_pre, y_pre = load_dataset_predict_3D(pre_file, i)
        x_pre = load_dataset_predict_3D(pre_file, i)
        x_pre = np.array(x_pre)
        x_pre = x_pre[:, :, :, :, np.newaxis]

        x_pre = model.predict(x_pre)
        x_pre *= 255.0

        x_pre = x_pre.reshape(split_num * split_num, pixel_size, pixel_size)
        if split_num == 1:
            x_pre = x_pre.reshape(pixel_size, pixel_size)
            x_pre_total = x_pre
        else:
            x_pre_total = pre_concatenate(x_pre)

        pre_dirpath = result_dirpath + r'\predict\\' + t_v + f'_predict_3D\\{pre_file}\\'
        pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i}'  # e=EPOCHS, # b=BATCH_SIZE
        os.makedirs(pre_dirpath + "save_npy", exist_ok=True)
        img_show(x_pre_total, pre_filepath)
        pre_npy_save(x_pre_total, pre_dirpath + f'save_npy\\pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}_{i}')


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,
                                            max_val=1.0, filter_size=11,
                                            filter_sigma=1.5, k1=0.01, k2=0.03))

if __name__ == "__main__":
    main()
