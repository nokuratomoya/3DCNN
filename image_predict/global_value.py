BATCH_SIZE = 16
dataset_total = 1200
dataset_num = int(dataset_total * 0.8)
# dataset_num = 1400

# train_data_date = "20230120"

##### 変更　#####
model_date = "timesize_2.5ms"
time_size = 150
xy_size = 3
pixel_size = 120
split_num = 1  # 分割サイズ　split_num*split_num分割される

# E_list = [1171, 1275, 1359, 1675, 1505]

# timesize_2.5ms = [10, 20, 30, 50, 100, 150]
E_list = [799, 1100, 1172, 1133]
# E = E_list[3]
E = 1801

# 学習に用いたCNN構造
# SID, 3D
model_dim = "3D"

# 復元させるスパイク画像の種類
# LNP, LNI, emulator, emulator_25
spike_data_name = "LNI"

# 一枚の画像に対し複数枚の出力をするときの変数
output3D = True  # 3D出力するかどうか
# pre_start_num:400, spike_data_num:50の場合, 351~400のデータを使って予測
pre_start_num = 400
pre_end_num = 600

# output=Falseの場合
# 刺激画像が始まる位置(教師画像は+1)
stim_head = 200
# stim_head = 214

# load_spike_date = "20240712"
# スパイクデータの保存先
# emulator
# predict_file_path =r"H:\train_data\20240711\0to1199"
# predict_file_path =r"H:\train_data\20240801\0to1199_2.5ms"
per = 1
a = 0.043 * per

gain_total = [1, 0.75, 0.5]
par_total = [1, 2, 4, 8]
gain_one = gain_total[0]
par_one = par_total[0]

nonlinear_gain = 24

# LNI
if spike_data_name == "LNI":
    predict_file_path = rf"E:\LNImodel\train_data\20241129\spatiotemporal_compare\gain{gain_one}_par{par_one}\gain{nonlinear_gain}_dt2.5\0to99"
    # predict_file_path = rf"E:\LNImodel\train_data\20241030\intensity_bias60_nodiff\gain24_dt2.5\0to99"

# LNP
elif spike_data_name == "LNP":
    predict_file_path = r"H:\G\LNPmodel\train_data\20240824\poisson_dt2e-05_dt2.5\0to99"

# emulator
elif spike_data_name == "emulator":
    predict_file_path = r"H:\train_data\20240711\0to1199"
# emulator_25
elif spike_data_name == "emulator_25":
    predict_file_path = r"H:\train_data\20240801\0to1199_2.5ms"

else:
    predict_file_path = ""

# 結果の保存先
# save_date = f"input_time_size_5ms\\150"


# emulator
if spike_data_name == "emulator" or "emulator_25":
    save_date = model_date
    save_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + save_date + fr"\result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{dataset_total}\predict\\"

# LNI
elif spike_data_name == "LNI" or "LNP":
    save_date = f"20241217//a_compare//gain{gain_one}_par{par_one}"
    save_path = fr"C:\Users\AIlab\labo\{spike_data_name}model\results\\" + save_date + r"\predict\\"


import datetime


def get_now():
    now = datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=9)))  # 日本時刻
    return now.strftime('%Y%m%d')  # yyyyMMdd
    # return now.strftime('%Y%m%d%H%M%S')  # yyyyMMddHHmmss
