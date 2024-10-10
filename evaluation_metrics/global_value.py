BATCH_SIZE = 16
dataset_total = 1200
dataset_num = int(dataset_total * 0.8)
# dataset_num = 32
# train_data_date = "20230120"
model_date = "20240903"

date = ""

a = 0.043

results_date = f"BMEiCON2024_diff\\a={a * 8}"
spike_data_name = "emulator"

# 結果の保存先
# results_predict_path = rf"E:\Users\AIlab\labo\{spike_data_name}model\results\\" + results_date + r"\predict\\"
results_predict_path = rf"C:\Users\AIlab\labo\3DCNN\results\20240903\result_kernel_150_3_3\predict\\"

# スパイクデータが入ったフォルダ
if spike_data_name == "LNI":
    eval_file_path = rf"H:\G\{spike_data_name}model\train_data\20240712\gain2_dt0.05\0to99"
elif spike_data_name == "LNP":
    eval_file_path = rf"H:\G\{spike_data_name}model\train_data\20240718\poisson_dt1.5e-05_dt0.05\0to99"
elif spike_data_name == "emulator":
    eval_file_path = rf"H:\train_data\20240711\0to1199"
elif spike_data_name == "emulator_25":
    eval_file_path = rf"H:\train_data\20240801\0to1199_2.5ms"

time_size = 150
# time_size = [40, 60, 80]
xy_size = 3
pixel_size = 120

E = 1156


start_num = 400
end_num = 600


import datetime


def get_now():
    now = datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=9)))  # 日本時刻
    return now.strftime('%Y%m%d')  # yyyyMMdd
    # return now.strftime('%Y%m%d%H%M%S')  # yyyyMMddHHmmss



