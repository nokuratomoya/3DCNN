from predict_img_func import predict_image, predict_any_image
from global_value import time_size, xy_size, E, model_date, get_now, predict_file_path, save_path, gain_total, par_total
import os
from natsort import natsorted
import csv


# E = 1794
# time_size = 100
# xy_size = 30


def main():
    global predict_file_path
    global save_path
    predict_file_paths = [rf"E:\LNImodel\train_data\20241008\gain{gain_total[1]}_par{par_total[2]}\gain2.5_dt2.5\0to99",
                          rf"E:\LNImodel\train_data\20241008\gain{gain_total[1]}_par{par_total[3]}\gain2.5_dt2.5\0to99",
                          rf"E:\LNImodel\train_data\20241008\gain{gain_total[2]}_par{par_total[0]}\gain2.5_dt2.5\0to99",
                          rf"E:\LNImodel\train_data\20241008\gain{gain_total[2]}_par{par_total[1]}\gain2.5_dt2.5\0to99",
                          rf"E:\LNImodel\train_data\20241008\gain{gain_total[2]}_par{par_total[2]}\gain2.5_dt2.5\0to99",
                          rf"E:\LNImodel\train_data\20241008\gain{gain_total[2]}_par{par_total[3]}\gain2.5_dt2.5\0to99"
                          ]

    save_paths = [rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[1]}_par{par_total[2]}\predict\\",
                  rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[1]}_par{par_total[3]}\predict\\",
                  rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[2]}_par{par_total[0]}\predict\\",
                  rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[2]}_par{par_total[1]}\predict\\",
                  rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[2]}_par{par_total[2]}\predict\\",
                  rf"C:\Users\AIlab\labo\LNImodel\results\20241008\gain{gain_total[2]}_par{par_total[3]}\predict\\",
                  ]

    print(f"{predict_file_path=}")
    print(f"{save_path=}")

    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # 保存先のディレクトリ

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

    # predict_file_path = r"G:\LNPmodel\train_data\20240619\poisson_dt7.5e-06\0to99"
    # predict_file_all = natsorted(os.listdir(predict_file_path))
    # pre_file = "img0"
    # i = 0
    # for pre_file in predict_file_all:
    #     print("predict_file_path: ", pre_file)
    #     predict_any_image(model_path, E, pre_file, save_path)
    #     # if i == 0:
    #     #     break
    for i in range(len(predict_file_paths)):
        predict_file_path = predict_file_paths[i]
        save_path = save_paths[i]
        # predict_file_all = natsorted(os.listdir(predict_file_paths[i]))
        pre_file = "img0"
        predict_any_image(model_path, E, pre_file, save_paths[i])


def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)

        # データを追加して保存する
        writer.writerow(save_data)

    file.close()


if __name__ == "__main__":
    main()
