from func_3DCNN import load_dataset, date, save_data_csv, save_data_txt, data_num, plot_history, hist_csv_save, \
    dirname_main, spike_data_num, spike_data_name, y_trains_save
from func_model import model_build, ssim_loss, SID_model_build
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import pathlib
import os
import time
import pickle
from natsort import natsorted

EPOCHS = 10000
BATCH_SIZE = 16

time_size = spike_data_num
xy_size = 3
filter_size = 4

train_dir = dirname_main

model_name = "3D"


def main():
    spike_data_num_all = [150]
    # spike_data_num_all = [200]
    global spike_data_num
    global time_size
    for spike in spike_data_num_all:
        spike_data_num = spike
        time_size = spike_data_num
        print(f"spike_data_num:{spike_data_num}"
              f"time_size:{time_size}")
    # global data_num
    # for d in data_num_all:
        # data_num = d
        # dataset準備
        start_load_time = time.perf_counter()
        x_trains, y_trains, unused_filename = load_dataset(data_num, spike_data_num)  # x_trains = [:, 100, 30, 30],y_trains = [:, 30, 30]
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
        os.makedirs(result_dirpath + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{data_num}\\', exist_ok=True)

        # y_trains_save(y_trains, result_dirpath)
        # 複数
        # kernel_size = [5, 10, 20]
        # filter_size = [3, 5, 7]

        save_data = ["filter", "time", "xy", "EPOCHS", spike_data_name]
        save_data_csv(save_data, result_dirpath + r"\EPOCHS_save.csv")
        if model_name == "3D":
            model3D(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size, result_dirpath)
        else:
            x_trains = x_trains.reshape(x_trains.shape[0] * x_trains.shape[1], x_trains.shape[2], x_trains.shape[3],
                                        x_trains.shape[4])
            # y_trains = y_trains.reshape(y_trains.shape[0], y_trains.shape[2], y_trains.shape[3], y_trains.shape[4])
            print(np.shape(x_trains), np.shape(y_trains))
            # y_trainsの複製
            y_trains = np.tile(y_trains, (1, spike_data_num, 1, 1, 1))
            print(np.shape(y_trains))
            pass
            y_trains = y_trains.reshape(y_trains.shape[0] * y_trains.shape[1], y_trains.shape[2], y_trains.shape[3],
                                        y_trains.shape[4])

            print(np.shape(x_trains), np.shape(y_trains))

            pass
            model2D_SID(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size, result_dirpath)


def model3D(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size, result_date_dirpath):
    print("debug datanum={}".format(data_num))
    start_train_time = time.perf_counter()
    input_shape = (len(x_trains[0]), len(x_trains[0][0]), len(x_trains[0][0][0]), 1)
    model = model_build(time_size, xy_size, filter_size, input_shape)

    checkpoint_path = result_date_dirpath + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{data_num}\\' + "cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="ssim_loss", min_delta=0.000, patience=100)
    # early_stopping = EarlyStopping(monitor="ssim_loss", min_delta=0.000, patience=50)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)


    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping, cp_callback])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = r"results\\" + date + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{data_num}\\'
    os.makedirs(result_dirpath, exist_ok=True)

    # historyの保存
    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    save_data_txt(unused_filename, save_unused_dir)
    save_data_csv(list(unused_filename),
                  result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    # train_dir = r"F:\train_data\20231129\stim400_cycle800ms"
    all_filename = natsorted(os.listdir(train_dir))
    all_filename = all_filename[:data_num]
    used_filename = set(all_filename) ^ set(unused_filename)
    used_filename = natsorted(used_filename)
    save_data_txt(used_filename, save_used_dir)
    save_data_csv(list(used_filename),
                  result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # かかった時間の保存
    save_elapsed_time = result_dirpath + f"model\\elapsed_time={round((end_train_time - start_train_time) / 60, 1)}min.txt"
    save_data_txt((end_train_time - start_train_time), save_elapsed_time)

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)

    # n_EPOCHSの保存

    # CSVファイルを追記モードで開く
    save_data = [filter_size, time_size, xy_size, n_EPOCHS]
    save_EPOCHS_csvname = result_date_dirpath + r"\EPOCHS_save.csv"
    save_data_csv(save_data, save_EPOCHS_csvname)


def model2D_SID(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size, result_date_dirpath):
    start_train_time = time.perf_counter()
    # input_shape = (len(x_trains[0]), len(x_trains[0][0]), 1)
    # model = model_build(time_size, xy_size, filter_size, input_shape)
    model = SID_model_build()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="ssim_loss", min_delta=0.000, patience=100)
    # early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)
    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = r"results\\" + date + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}\\'
    os.makedirs(result_dirpath, exist_ok=True)

    # historyの保存
    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    save_data_txt(unused_filename, save_unused_dir)
    save_data_csv(list(unused_filename),
                  result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    # train_dir = r"F:\train_data\20231129\stim400_cycle800ms"
    all_filename = natsorted(os.listdir(train_dir))
    all_filename = all_filename[:data_num]
    used_filename = set(all_filename) ^ set(unused_filename)
    used_filename = natsorted(used_filename)
    save_data_txt(used_filename, save_used_dir)
    save_data_csv(list(used_filename),
                  result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv")

    # かかった時間の保存
    save_elapsed_time = result_dirpath + f"model\\elapsed_time={round((end_train_time - start_train_time) / 60, 1)}min.txt"
    save_data_txt((end_train_time - start_train_time), save_elapsed_time)

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)

    # n_EPOCHSの保存

    # CSVファイルを追記モードで開く
    save_data = [filter_size, time_size, xy_size, n_EPOCHS]
    save_EPOCHS_csvname = result_date_dirpath + r"\EPOCHS_save.csv"
    save_data_csv(save_data, save_EPOCHS_csvname)


if __name__ == '__main__':
    main()
