BATCH_SIZE = 16
dataset_total = 1200
dataset_num = int(dataset_total * 0.8)
# dataset_num = 1400

# train_data_date = "20230120"

model_date = "20240903"
time_size = 150
xy_size = 3
pixel_size = 120
split_num = 1  # 分割サイズ　split_num*split_num分割される

# 一枚の画像に対し複数枚の出力をするときの変数
output3D = True  # 3D出力するかどうか
# pre_start_num:400, spike_data_num:50の場合, 351~400のデータを使って予測
pre_start_num = 200
pre_end_num = 350

# 刺激画像が始まる位置(教師画像は+1)
# stim_head  =201
stim_head = 214

# SID, 3D
model_dim = "3D"
# LNP, emulator, emulator_25
spike_data_name = "LNI"

# load_spike_date = "20240712"
# スパイクデータの保存先
# emulator
# predict_file_path =r"H:\train_data\20240711\0to1199"
# predict_file_path =r"H:\train_data\20240801\0to1199_2.5ms"
per = 8
a = 0.043 * per

# LNI
predict_file_path = rf"H:\G\LNImodel\train_data\20240712\gain2_dt0.05\0to99"
# predict_file_path = rf"H:\G\LNImodel\train_data\20240911_diff\a={a}\gain2.5_dt2.5\0to99"
# LNP
# predict_file_path = r"H:\G\LNPmodel\train_data\20240824\poisson_dt2e-05_dt2.5\0to99"

# 結果の保存先
# save_date = f"input_time_size_5ms\\150"
save_date = "20240911/predict_20240903/gain2"

# emulator
# save_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + save_date + fr"\result_kernel_{time_size}_{xy_size}_{xy_size}\predict\\"

# LNI
save_path = fr"C:\Users\AIlab\labo\{spike_data_name}model\results\\" + save_date + r"\predict\\"

# EPOCHS
# E = (
# [time1, time2, time3], xy1
# [time1, time2, time3], xy2
# [time1, time2, time3] xy3
# )
# E = ([1057, 1070, 930], [864, 1044, 905], [728, 1059, 755])  # 20230704
# E = ([1499, 1048, 1567], [1075, 840, 613], [1000, 925, 904])  # 20230707
# E = ([623, 853, 676], [1259, 845, 1160], [922, 835, 115])  # 20230712
E = 1156

import datetime


def get_now():
    now = datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=9)))  # 日本時刻
    return now.strftime('%Y%m%d')  # yyyyMMdd
    # return now.strftime('%Y%m%d%H%M%S')  # yyyyMMddHHmmss
