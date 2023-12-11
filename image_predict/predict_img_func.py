# predict_imageをxy_sizeを変えて複数回す用
# EPOCHSだけ自分で設定する必要あり

from tensorflow.keras.models import load_model
from func_3DCNN_predict import load_dataset_predict, split_num, img_show, pre_concatenate, pre_npy_save
import numpy as np
import os
import pickle
from global_value import BATCH_SIZE, dataset_num, model_date
from natsort import natsorted


def predict_calc_save(pre_file, t_v, result_dirpath, EPOCHS, model):

    # pre_file = '221017_data'
    x_pre = load_dataset_predict(pre_file)
    x_pre = np.array(x_pre)
    x_pre = x_pre[:, :, :, :, np.newaxis]

    x_pre = model.predict(x_pre)
    x_pre *= 255.0
    x_pre = x_pre.reshape(split_num * split_num, 30, 30)
    x_pre_total = pre_concatenate(x_pre)
    # y_pre_total = pre_concatenate(y_pre)

    # pre_dirpath = date + r'\predict\\' + t_v + '_predict\\'  # k_20, train
    pre_dirpath = result_dirpath + r'\predict\\' + t_v + '_predict\\'
    pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'  # e=EPOCHS, # b=BATCH_SIZE
    os.makedirs(pre_dirpath + "save_npy", exist_ok=True)
    img_show(x_pre_total, pre_filepath)
    pre_npy_save(x_pre_total, pre_dirpath + f'save_npy\\pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}')


def predict_image(model_name, EPOCHS):
    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + r'\\' + model_name + '\\'  # k_20
    model_h5 = result_dirpath + "model" + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    model = load_model(model_h5)

    # unused_data(テストデータ)
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = natsorted(pickle.load(f))
    f.close()
    for i in unused_filename:
        predict_calc_save(i, "test", result_dirpath, EPOCHS, model)
        print(f"finished:{i}")

    # used(訓練データ)
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = natsorted(pickle.load(f))
    f.close()
    for i in used_filename:
        predict_calc_save(i, "train", result_dirpath, EPOCHS, model)
        print(f"finished:{i}")



