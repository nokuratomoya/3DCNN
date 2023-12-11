# 相互情報量(MI)の計算
# ファイルのimport
from calcMI_func import calc_test_MI, calc_train_MI
from global_value import xy_size, time_size, E


def main():
    # for i, time in enumerate(time_size):
    #     for j, xy in enumerate(xy_size):
    #         model_name = f"result_kernel_{time}_{xy}_{xy}"
    #         # E = ([1057, 1070, 930], [864, 1044, 905], [728, 1059, 755])
    #         calc_test_MI(model_name, E[i][j])
    #         calc_train_MI(model_name, E[i][j])

    model_name = f"result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}"
    calc_test_MI(model_name, E)
    calc_train_MI(model_name, E)


if __name__ == "__main__":
    main()
