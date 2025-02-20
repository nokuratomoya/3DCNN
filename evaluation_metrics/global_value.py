index = -1

BATCH_SIZE = 16
datanum_list = [5, 10, 15, 20, 40, 80, 100, 200, 400, 1200]
dataset_total = datanum_list[-1]
# dataset_total = 1200
dataset_num = int(dataset_total * 0.8)
# dataset_num = 32
# train_data_date = "20230120"
model_date = "timesize_2.5ms"
# model_date = "20240820"
timesize_list = [10, 20, 30, 40, 50, 80, 100, 150]
time_size = timesize_list[index]
# time_size = [40, 60, 80]
xy_size = 3
pixel_size = 120
# E = 1628

# dataset_size = [5, 10, 15, 20, 40, 80, 100, 200, 400, 1200]
# E_list = [2817, 6144, 4814, 4147, 1901, 2442, 3147, 1811, 1135, 1386]

# timesize_5ms = [10, 20, 30, 50, 100, 150]
# E_list = [1157, 1680, 1119, 1078, 1386, 1156]

# timesize_2.5ms = [10, 20, 40, 80, 100]
E_list = [799, 1100, 1172, 1133, 1628]
# E = E_list[index]
E = 1801


# spike:emulator_2.5ms
if model_date == "20240820":
    E = 1628
# spike:emulator_5ms
if model_date == "20240810":
    E = 1386


date = ""

output3D = True

a = 0.043 * 1
gain_total = [1, 0.75, 0.5]
par_total = [1, 2, 4, 8]
gain_one = gain_total[0]
par_one = par_total[0]
nonlinear_gain = 24

results_date = rf"20241217\\a_compare"
spike_data_name = "LNI"

# 結果の保存先
results_predict_path = rf"C:\Users\AIlab\labo\{spike_data_name}model\results\\" + results_date + r"\predict\\"
if spike_data_name == "emulator" or "emulator_25":
    # データセットがフォルダ名についている場合
    # results_predict_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + fr"\result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{dataset_total}\predict\\"
    # ついてない場合
    results_predict_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + model_date + fr"\result_kernel_{time_size}_{xy_size}_{xy_size}\predict\\"


# スパイクデータが入ったフォルダ
if spike_data_name == "LNI":
    # eval_file_path = rf"H:\G\{spike_data_name}model\train_data\20240712\gain2_dt0.05\0to99"
    eval_file_path = rf"E:\LNImodel\train_data\20241008\gain{gain_one}_par{par_one}\gain2.5_dt2.5\0to99"
elif spike_data_name == "LNP":
    eval_file_path = rf"H:\G\{spike_data_name}model\train_data\20240718\poisson_dt1.5e-05_dt0.05\0to99"
elif spike_data_name == "emulator":
    eval_file_path = rf"H:\train_data\20240711\0to1199"
elif spike_data_name == "emulator_25":
    eval_file_path = rf"H:\train_data\20240801\0to1199_2.5ms"


start_num = 400
end_num = 600

import datetime

def get_now():
    now = datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=9)))  # 日本時刻
    return now.strftime('%Y%m%d')  # yyyyMMdd
    # return now.strftime('%Y%m%d%H%M%S')  # yyyyMMddHHmmss



