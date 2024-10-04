from predict_img_func import predict_image, predict_any_image
from global_value import time_size, xy_size, E, model_date, get_now, predict_file_path, save_path
import os
from natsort import natsorted
import csv


# E = 1794
# time_size = 100
# xy_size = 30

def main():
    predict_file_paths = [r"H:\G\LNImodel\train_data\eguchi\20240714_a=0.005375\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\eguchi\20240713_a=0.01075\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\eguchi\20240712_a=0.0215\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\nakamura\20240719\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\nakamura\20240712\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\nakamura\20240717\gain2_dt0.05\0to99",
                          r"H:\G\LNImodel\train_data\nakamura\20240718\gain2_dt0.05\0to99"]

    save_paths = [r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.005375\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.01075\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.0215\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.043\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.086\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.172\predict\\",
                  r"C:\Users\AIlab\labo\LNImodel\results\SII2025\a=0.344\predict\\"
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
    predict_file_all = natsorted(os.listdir(predict_file_path))
    # pre_file = "img0"
    i = 0
    for pre_file in predict_file_all:
        print("predict_file_path: ", pre_file)
        predict_any_image(model_path, E, pre_file, save_path)
        # if i == 0:
        #     break


def save_data_csv(save_data, save_dir_name):
    # CSVファイルを追記モードで開く
    with open(save_dir_name, 'a', newline='') as file:
        writer = csv.writer(file)

        # データを追加して保存する
        writer.writerow(save_data)

    file.close()


if __name__ == "__main__":
    main()
