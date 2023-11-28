from func_3Dver import load_dataset, model_build, plot_history, img_show, \
    img_shows, load_data_predict, split_num, date, hist_csv_save, model_build_kernel_20, model_build_kernel_100, \
    model_build_kernel_100_less, model_build_kernel_change, save_data_txt, save_data_csv
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pathlib
import os
import time
import pickle


EPOCHS = 10000
BATCH_SIZE = 16


# time_size = [40, 60, 80]
time_size = [100]
xy_size = [3]
# xy_size = [7]
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
    y_trains /= 255.0
    x_trains, y_trains = x_trains.astype('float32'), y_trains.astype('float32')
    #
    print(np.shape(x_trains), np.shape(y_trains))
    start_train_time = time.perf_counter()

    result_dirpath = date
    os.makedirs(result_dirpath, exist_ok=True)

    # model_kernel10(x_trains, y_trains, unused_filename, dataset_num)

    # model_kernel20(x_trains, y_trains, unused_filename, dataset_num)

    # model_kernel100(x_trains, y_trains, unused_filename, dataset_num)

    # model_kernel_100_less(x_trains, y_trains, unused_filename, dataset_num)

    # 複数
    # kernel_size = [5, 10, 20]
    # filter_size = [3, 5, 7]

    save_data = ["filter", "time", "xy", "EPOCHS"]
    save_data_csv(save_data, r"C:\Users\AIlab\labo\3DCNN\\" + date + r"\EPOCHS_save.csv")
    for i in time_size:
        for j in xy_size:
            model_kernel_change(x_trains, y_trains, unused_filename, dataset_num, i, j, filter_size)


def model_kernel10(x_trains, y_trains, unused_filename, dataset_num):
    start_train_time = time.perf_counter()
    model = model_build()
    # model = model_build_kernel_20()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + r'\results\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\validation_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(save_unused_dir, 'wb')
    pickle.dump(unused_filename, f)
    f.close()

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)


def model_kernel20(x_trains, y_trains, unused_filename, dataset_num):
    start_train_time = time.perf_counter()
    model = model_build_kernel_20()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + r'\results_k20\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\validation_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(save_unused_dir, 'wb')
    pickle.dump(unused_filename, f)
    f.close()

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)


def model_kernel100(x_trains, y_trains, unused_filename, dataset_num):
    start_train_time = time.perf_counter()
    model = model_build_kernel_100()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + r'\results_k100\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\validation_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(save_unused_dir, 'wb')
    pickle.dump(unused_filename, f)
    f.close()

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    train_dir = "train_data\\20230120"
    all_filename = os.listdir(train_dir)
    used_filename = set(all_filename) ^ set(unused_filename)
    f = open(save_used_dir, 'wb')
    pickle.dump(used_filename, f)
    f.close()

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)


def model_kernel_100_less(x_trains, y_trains, unused_filename, dataset_num):
    start_train_time = time.perf_counter()
    model = model_build_kernel_100_less()

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + r'\result\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    f = open(save_unused_dir, 'wb')
    pickle.dump(unused_filename, f)
    f.close()

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    train_dir = "train_data\\20230120"
    all_filename = os.listdir(train_dir)
    used_filename = set(all_filename) ^ set(unused_filename)
    f = open(save_used_dir, 'wb')
    pickle.dump(used_filename, f)
    f.close()

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)


def model_kernel_change(x_trains, y_trains, unused_filename, dataset_num, time_size, xy_size, filter_size):
    start_train_time = time.perf_counter()
    model = model_build_kernel_change(time_size, xy_size, filter_size)

    # 打ち切り設定
    early_stopping = EarlyStopping(monitor="loss", min_delta=0.000, patience=100)

    # history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    history = model.fit(x_trains, y_trains, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stopping])

    end_train_time = time.perf_counter()
    print(f"training_time:{(end_train_time - start_train_time) / 60}min")

    # resultフォルダ作成
    result_dirpath = date + f'\\result_kernel_{time_size}_{xy_size}_{xy_size}\\'
    os.makedirs(result_dirpath, exist_ok=True)

    n_EPOCHS = len(history.history["loss"])

    result_filepath = result_dirpath + f'result_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.png'
    model_savepath = result_dirpath + f"model\\model_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.h5"
    history_filepath = result_dirpath + f"model\\history_dataset={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.csv"

    model.save(model_savepath)

    # unused_filename(評価データ)の保存
    save_unused_dir = result_dirpath + f"model\\test_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    save_data_txt(unused_filename, save_unused_dir)

    # used_file
    save_used_dir = result_dirpath + f"model\\train_data={dataset_num}_e={n_EPOCHS}_b={BATCH_SIZE}.txt"
    train_dir = "train_data\\20230120"
    all_filename = os.listdir(train_dir)
    used_filename = set(all_filename) ^ set(unused_filename)
    save_data_txt(used_filename, save_used_dir)


    # かかった時間の保存
    save_elapsed_time = result_dirpath + f"model\\elapsed_time={round((end_train_time - start_train_time) / 60, 1)}min.txt"
    save_data_txt((end_train_time - start_train_time), save_elapsed_time)

    # plot_history(history)
    plot_history(history, result_filepath)
    hist_csv_save(history, history_filepath)

    # n_EPOCHSの保存

    # CSVファイルを追記モードで開く
    save_data = [filter_size, time_size, xy_size, n_EPOCHS]
    save_EPOCHS_csvname = r"C:\Users\AIlab\labo\3DCNN\\" + date + r"\EPOCHS_save.csv"
    save_data_csv(save_data, save_EPOCHS_csvname)


if __name__ == '__main__':
    main()
