from tensorflow.keras.models import load_model
from func_3Dver import load_data_predict, split_num, img_show, img_compare, pre_concatenate
import numpy as np
import os
import pickle
from global_value import EPOCHS, BATCH_SIZE, dataset_num, date, model_name
"""
# ここで指定したmodelで復元を行う
########################

EPOCHS = 10001
BATCH_SIZE = 16
dataset_num = 58 * 100
date = "20230120"

########################

# 復元する画像
pre_file = 'adler_data'
"""


def predict_image(pre_file, t_v, result_dirpath):
    dirname = result_dirpath + 'model'  # k_20
    model_h5 = dirname + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'

    model = load_model(model_h5)

    # pre_file = '221017_data'
    x_pre, y_pre = load_data_predict(pre_file)
    x_pre = np.array(x_pre)
    x_pre = x_pre[:, :, :, :, np.newaxis]

    x_pre = model.predict(x_pre)
    x_pre *= 255.0
    x_pre = x_pre.reshape(split_num * split_num, 30, 30)
    x_pre_total = pre_concatenate(x_pre)
    y_pre_total = pre_concatenate(y_pre)

    # pre_dirpath = date + r'\predict\\' + t_v + '_predict\\'  # k_20, train
    pre_dirpath = result_dirpath + r'\predict\\' + t_v + '_predict\\'
    os.makedirs(pre_dirpath, exist_ok=True)
    pre_filepath = pre_dirpath + f'pre_{pre_file}_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'  # e=EPOCHS, # b=BATCH_SIZE
    img_show(x_pre_total, pre_filepath)

    # 相互情報量のほうで実行
    """
    pre_compare = [x_pre_total, y_pre_total]
    img_compare(pre_compare, pre_filepath, pre_dirpath)
    """


def main():
    # すべてのデータにおいて推測
    # unused(value)
    result_dirpath = date + r'\\' + model_name + '\\'  # k_20
    load_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_unused_dir, 'rb')
    unused_filename = pickle.load(f)
    f.close()
    for i in unused_filename:
        predict_image(i, "test", result_dirpath)
        print(f"finished:{i}")
    # used(train)
    load_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(load_used_dir, 'rb')
    used_filename = pickle.load(f)
    f.close()
    for i in used_filename:
        predict_image(i, "train", result_dirpath)
        print(f"finished:{i}")


if __name__ == "__main__":
    main()
