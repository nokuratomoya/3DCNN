BATCH_SIZE = 16
dataset_num = 960
train_data_date = "20230120"

date = "20240111"
time_size = [20]
# time_size = [40, 60, 80]
xy_size = [3]

# EPOCHS
# E = (
# [time1, time2, time3], xy1
# [time1, time2, time3], xy2
# [time1, time2, time3] xy3
# )
# E = ([1057, 1070, 930], [864, 1044, 905], [728, 1059, 755])  # 20230704
# E = ([1499, 1048, 1567], [1075, 840, 613], [1000, 925, 904])  # 20230707
# E = ([623, 853, 676], [1259, 845, 1160], [922, 835, 115])  # 20230712
E = 1179

import datetime


def get_now():
    now = datetime.datetime.now(datetime.timezone(
        datetime.timedelta(hours=9)))  # 日本時刻
    return now.strftime('%Y%m%d')  # yyyyMMdd
    # return now.strftime('%Y%m%d%H%M%S')  # yyyyMMddHHmmss
