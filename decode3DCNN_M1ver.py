from func_3DCNN import load_dataset, date, save_data_csv, save_data_txt, plot_history, hist_csv_save
from func_model import model_build
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pathlib
import os
import time
import pickle

EPOCHS = 10000
BATCH_SIZE = 16

time_size = 100
xy_size = 3
filter_size = 4


def main():
    # dataset準備
    start_load_time = time.perf_counter()
    x_trains, y_trains, unused_filename = load_dataset()  # x_trains = [:, 100, 30, 30],y_trains = [:, 30, 30]
    end_load_time = time.perf_counter()
    print(f"load_time:{end_load_time - start_load_time}s")
    dataset_num = len(x_trains)
    # reshape
    x_trains, y_trains = np.array(x_trains), np.array(y_trains)
    x_trains, y_trains = x_trains[:, :, :, :, np.newaxis], y_trains[:, np.newaxis, :, :, np.newaxis]  # np.reshape

    x_trains, y_trains = x_trains.astype('float32'), y_trains.astype('float32')
    y_trains /= 255.0

    # x_trains:[dataset_num, spike_data_num, pixel, pixel, 1]
    # y_trains:[dataset_num, 1, pixel, pixel, 1]

    print(np.shape(x_trains), np.shape(y_trains))

    result_dirpath = r"C:\Users\AIlab\labo\3DCNN\results\\" + date
    os.makedirs(result_dirpath, exist_ok=True)

    # 複数
    # kernel_size = [5, 10, 20]
    # filter_size = [3, 5, 7]

    save_data = ["filter", "time", "xy", "EPOCHS"]
    save_data_csv(save_data, result_dirpath + r"\EPOCHS_save.csv")
    model3D(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size)
    # for i in time_size:
    #     for j in xy_size:
    #         model_kernel_change(x_trains, y_trains, unused_filename, dataset_num, i, j, filter_size)


def model3D(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size):
    start_train_time = time.perf_counter()
    input_shape = (len(x_trains[0]), len(x_trains[0][0]), len(x_trains[0][0][0]), 1)
    model = model_build(time_size, xy_size, filter_size, input_shape)

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="MeanSquaredError", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = r"results\\" + date + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    save_data_txt(unused_filename, save_unused_dir)
    save_data_csv(list(unused_filename), result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    train_dir = r"F:\train_data\20231129\stim400_cycle800ms"
    all_filename = os.listdir(train_dir)
    used_filename = set(all_filename) ^ set(unused_filename)
    save_data_txt(used_filename, save_used_dir)
    save_data_csv(list(used_filename), result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # かかった時間の保存
    save_elapsed_time = result_dirpath + f"model\\elapsed_time={round((end_train_time - start_train_time) / 60, 1)}min.txt"
    save_data_txt((end_train_time - start_train_time), save_elapsed_time)

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)

    # n_EPOCHSの保存

    # CSVファイルを追記モードで開く
    save_data = [filter_size, time_size, xy_size, n_EPOCHS]
    save_EPOCHS_csvname = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + r"\EPOCHS_save.csv"
    save_data_csv(save_data, save_EPOCHS_csvname)


if __name__ == '__main__':
    main()
