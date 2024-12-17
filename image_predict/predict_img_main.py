from predict_img_func import predict_image, predict_any_image
from global_value import time_size, xy_size, E, model_date, get_now, gain_total, par_total, predict_file_path, save_path, output3D
import os
from natsort import natsorted
import csv
import itertools

# E = 1794
# time_size = 100
# xy_size = 30

# predict_file_path = ""
# save_path = ""

def main():

    # print(f"{predict_file_path=}")
    # print(f"{save_path=}")
    os.makedirs(save_path, exist_ok=True)
    if output3D:
        save_csv_filename = save_path + "trained_model_3D.csv"
    else:
        save_csv_filename = save_path + "trained_model.csv"
    # model_dateの保存
    save_data_csv(["model_date",
                   "time_size",
                   "xy_size",
                   "EPOCHS",
                   "predict_file_path",
                   ], save_csv_filename)

    save_data_csv([model_date,
                   time_size,
                   xy_size,
                   E,
                   predict_file_path,
                   ], save_csv_filename)

    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # 保存先のディレクトリ

    # predict_file_path = r"G:\LNPmodel\train_data\20240619\poisson_dt7.5e-06\0to99"
    predict_file_all = natsorted(os.listdir(predict_file_path))
    # pre_file = "img0"
    # i = 0
    #
    for i, pre_file in enumerate(predict_file_all):
        print("predict_file_path: ", pre_file)
        predict_any_image(model_path, E, pre_file, save_path, predict_file_path)
        # if i == 0:
        #     break
    """


    ######################
    # loop
    # predict_file_paths = [rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[1]}_par{par_total[0]}\gain2.5_dt2.5\\0to99",
    #                       rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[1]}_par{par_total[1]}\gain2.5_dt2.5\\0to99",
    #                       rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[0]}_par{par_total[0]}\gain2.5_dt2.5\\0to99",
    #                       rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[0]}_par{par_total[1]}\gain2.5_dt2.5\\0to99",
    #                       rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[0]}_par{par_total[2]}\gain2.5_dt2.5\\0to99",
    #                       rf"E:\LNImodel\\train_data\\20241008\gain{gain_total[0]}_par{par_total[3]}\gain2.5_dt2.5\\0to99"
    #                       ]
    #
    # save_paths = [rf"C:\\Users\AIlab\labo\LNImodel\\results\20241011\gain{gain_total[1]}_par{par_total[0]}\predict\\",
    #               rf"C:\\Users\AIlab\labo\LNImodel\\results\20241011\gain{gain_total[1]}_par{par_total[1]}\predict\\",
    #               rf"C:\\Users\AIlab\labo\LNImodel\\results\20241011\gain{gain_total[0]}_par{par_total[0]}\predict\\",
    #               rf"C:\\Users\AIlab\labo\LNImodel\\results\20241011\gain{gain_total[0]}_par{par_total[1]}\predict\\",
    #               rf"C:\\Users\AIlab\labo\LNImodel\\results\\20241011\gain{gain_total[0]}_par{par_total[2]}\predict\\",
    #               rf"C:\\Users\AIlab\labo\LNImodel\\results\20241011\gain{gain_total[0]}_par{par_total[3]}\predict\\",
    #               ]

    global predict_file_path
    global save_path
    spatial_filter_gain_loop = [1, 0.75, 0.5]
    temporal_filter_par_loop = [1, 2, 4, 8]
    a_list = [0.005375 , 0.01075, 0.0215, 0.043, 0.086, 0.172, 0.344 ]
    loop_list = list(itertools.product(spatial_filter_gain_loop, temporal_filter_par_loop))

    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"
    # for gain, par in loop_list:
    for a in a_list:

        predict_file_path = rf"E:\LNImodel\train_data\20241030\intensity_bios60\a={a}\gain24_dt2.5\0to99"
        save_path = rf"C:\\Users\AIlab\labo\LNImodel\\results\\20241030\intensity_bios60/a={a}\predict\\"
        # 保存先のディレクトリ

        os.makedirs(save_path, exist_ok=True)

        print(f"{predict_file_path=}")
        print(f"{save_path=}")
        predict_file_all = natsorted(os.listdir(predict_file_path))
        print(len(predict_file_all))
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

        pre_file = "img0"
        print("predict_file_path: ", pre_file)
        predict_any_image(model_path, E, pre_file, save_path, predict_file_path)
        # for pre_file in predict_file_all:
        #     print("predict_file_path: ", pre_file)
        #     predict_any_image(model_path, E, pre_file, save_path)

    #######################
    """


def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)

        # データを追加して保存する
        writer.writerow(save_data)

    file.close()


if __name__ == "__main__":
    main()
