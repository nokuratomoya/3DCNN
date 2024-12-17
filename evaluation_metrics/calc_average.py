import os
import numpy as np
import pandas as pd
from global_value import results_date, time_size, xy_size, dataset_num, E, BATCH_SIZE, results_predict_path, start_num, \
    end_num
import itertools
from natsort import natsorted


def main():
    # MI_NCC_SSIM_average()

    # results_path = r"C:\Users\AIlab\labo\LNPmodel\results\\" + results_date + rf"\predict\\"

    # total_average(results_predict_path, SSIM=True, NCC=False, total=False)

    SSIM_all_average(results_predict_path)
    # NCCspearman_all_average(results_predict_path)


    ##########
    # # loop
    # spatial_filter_gain_loop = [1, 0.75, 0.5]
    # temporal_filter_par_loop = [1, 2, 4, 8]
    # loop_list = list(itertools.product(spatial_filter_gain_loop, temporal_filter_par_loop))
    # for gain, par in loop_list:
    #     results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\\20241017\maxGP1\gain{gain}_par{par}\\predict\\"
    #     SSIM_all_average(results_predict_path)
        # NCCspearman_all_average


def MI_NCC_SSIM_average():
    results_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + results_date + rf"\result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}\predict\\"

    MI_train_path = results_path + f"mutual_info/mutual_info={dataset_num}_e={E}_b={BATCH_SIZE}_train.csv"
    MI_test_path = results_path + f"mutual_info/mutual_info={dataset_num}_e={E}_b={BATCH_SIZE}_test.csv"
    # NCC_path = results_path + f"NCC/dataset={dataset_num}_e={E}_b={BATCH_SIZE}/NCC_total.csv"
    NCC_spearman_path = results_path + f"NCC/dataset={dataset_num}_e={E}_b={BATCH_SIZE}/spearman_total.csv"
    SSIM_path = results_path + f"SSIM/SSIM.csv"
    print(NCC_spearman_path)
    df_MI_train = read_csv_df_MI(MI_train_path)
    df_MI_test = read_csv_df_MI(MI_test_path)
    # df_NCC = read_csv_df(NCC_path)
    col_names = ['c{0:02d}'.format(i) for i in range(1200)]
    df_NCC_spearman = read_csv_df(NCC_spearman_path, col_names)
    df_SSIM = read_csv_df(SSIM_path, col_names)

    print(f"MI_train:{df_MI_train.mean(axis=1)[0]},{df_MI_train.mean(axis=1)[1]}")
    print(f"MI_test:{df_MI_test.mean(axis=1)[0]},{df_MI_test.mean(axis=1)[1]}")
    # print(f"NCC:{df_NCC.mean(axis=1)[0]},{df_NCC.mean(axis=1)[1]}")
    print(f"NCC_spearman:{df_NCC_spearman.mean(axis=1)[0]},{df_NCC_spearman.mean(axis=1)[1]}")
    print(f"SSIM:{df_SSIM.mean(axis=1)[0]},{df_SSIM.mean(axis=1)[1]}")

    save_ave_csv_path = results_path + f"average.csv"
    df_ave = pd.DataFrame(
        [[df_MI_train.mean(axis=1)[0], df_MI_train.mean(axis=1)[1], df_MI_test.mean(axis=1)[0],
          df_MI_test.mean(axis=1)[1], df_NCC_spearman.mean(axis=1)[0], df_NCC_spearman.mean(axis=1)[1],
          df_SSIM.mean(axis=1)[0], df_SSIM.mean(axis=1)[1]]],
        columns=["MI_train_ave", "MI_train", "MI_test_ave", "MI_test", "NCC_spearman_train", "NCC_spearman_test",
                 "SSIM_train", "SSIM_test"])

    print(df_ave)
    df_ave.to_csv(save_ave_csv_path, index=False)


def total_average(results_path, SSIM=False, NCC=False, total=False):
    NCC_spearman_path = results_path + f"NCC_spearman/spearman_NCC.csv"
    SSIM_path = results_path + f"SSIM/SSIM.csv"
    col_names = ['c{0:02d}'.format(i) for i in range(1200)]
    col_names_total = ['c{0:02d}'.format(i) for i in range(100)]

    save_data_value = []
    columns = []
    if NCC:
        df_NCC_spearman = read_csv_df(NCC_spearman_path, col_names)
        print(f"NCC_spearman:{df_NCC_spearman.mean(axis=1)[0]},{df_NCC_spearman.mean(axis=1)[1]}")
        save_data_value.append(df_NCC_spearman.mean(axis=1)[0])
        save_data_value.append(df_NCC_spearman.mean(axis=1)[1])
        columns.append("NCC_spearman_train")
        columns.append("NCC_spearman_test")
        if total:
            NCC_spearman_total_path = results_path + f"NCC_spearman/spearman_NCC_total.csv"
            df_NCC_spearman_total = read_csv_df_total(NCC_spearman_total_path, col_names_total)
            print(f"NCC_spearman_total:{df_NCC_spearman_total.mean(axis=1)[0]}")
            save_data_value.append(df_NCC_spearman_total.mean(axis=1)[0])
            columns.append("NCC_spearman_total")

    if SSIM:
        df_SSIM = read_csv_df(SSIM_path, col_names)
        print(f"SSIM:{df_SSIM.mean(axis=1)[0]},{df_SSIM.mean(axis=1)[1]}")
        save_data_value.append(df_SSIM.mean(axis=1)[0])
        save_data_value.append(df_SSIM.mean(axis=1)[1])
        columns.append("SSIM_train")
        columns.append("SSIM_test")
        if total:
            SSIM_total_path = results_path + f"SSIM/SSIM_total.csv"
            df_SSIM_total = read_csv_df_total(SSIM_total_path, col_names_total)
            print(f"SSIM_total:{df_SSIM_total.mean(axis=1)[0]}")
            save_data_value.append(df_SSIM_total.mean(axis=1)[0])
            columns.append("SSIM_total")

    save_ave_csv_path = results_path + f"average.csv"
    df_ave = pd.DataFrame(
        [save_data_value],
        columns=columns)

    print(df_ave)
    df_ave.to_csv(save_ave_csv_path, index=False)


def SSIM_all_average(results_path):
    SSIM_path = results_path + f"\\SSIM_3D\\"
    filename_all = natsorted(os.listdir(SSIM_path))

    col_names = ['c{0:02d}'.format(i) for i in range(end_num - start_num + 1)]
    df_SSIM_all = pd.DataFrame(columns=col_names)
    for filename in filename_all:
        SSIM_one_path = SSIM_path + filename
        df_SSIM_one = read_csv_df_total(SSIM_one_path, col_names)
        df_SSIM_all = pd.concat([df_SSIM_all, df_SSIM_one], ignore_index=True)

    save_ave_csv_path = results_path + f"SSIM_total"
    os.makedirs(save_ave_csv_path, exist_ok=True)
    df_ave = df_SSIM_all.mean(axis=0)
    df_std = df_SSIM_all.std(axis=0)
    df_sem = df_SSIM_all.sem(axis=0)

    df_ave_np = np.array(df_ave)
    df_std_np = np.array(df_std)
    df_sem_np = np.array(df_sem)
    np.savetxt(save_ave_csv_path + "\\SSIM_3D_average.csv", df_ave_np.transpose(), delimiter=",")
    np.savetxt(save_ave_csv_path + "\\SSIM_3D_std.csv", df_std_np.transpose(), delimiter=",")
    np.savetxt(save_ave_csv_path + "\\SSIM_3D_sem.csv", df_sem_np.transpose(), delimiter=",")
    # df_ave.to_csv(save_ave_csv_path + "\\SSIM_3D_average.csv", index=False, header=False)


def NCCspearman_all_average(results_path):
    NCC_path = results_path + f"\\NCCspearman_3D\\"
    filename_all = natsorted(os.listdir(NCC_path))

    col_names = ['c{0:02d}'.format(i) for i in range(end_num - start_num + 1)]
    df_NCC_all = pd.DataFrame(columns=col_names)
    for filename in filename_all:
        NCC_one_path = NCC_path + filename
        df_NCC_one = read_csv_df_total(NCC_one_path, col_names)
        df_NCC_all = pd.concat([df_NCC_all, df_NCC_one], ignore_index=True)

    save_ave_csv_path = results_path + f"NCCspearman_total"
    os.makedirs(save_ave_csv_path, exist_ok=True)
    df_ave = df_NCC_all.mean(axis=0)
    df_std = df_NCC_all.std(axis=0)
    df_sem = df_NCC_all.sem(axis=0)

    df_ave_np = np.array(df_ave)
    df_std_np = np.array(df_std)
    df_sem_np = np.array(df_sem)
    np.savetxt(save_ave_csv_path + "\\NCCspearman_3D_average.csv", df_ave_np.transpose(), delimiter=",")
    np.savetxt(save_ave_csv_path + "\\NCCspearman_3D_std.csv", df_std_np.transpose(), delimiter=",")
    np.savetxt(save_ave_csv_path + "\\NCCspearman_3D_sem.csv", df_sem_np.transpose(), delimiter=",")

def read_csv_df_MI(path):
    df = pd.read_csv(path, encoding="shift_jis", index_col=0, skiprows=0)
    return df


def read_csv_df(path, col_names):
    df = pd.read_csv(path, header=None, skiprows=[0, 2], names=col_names)
    return df


def read_csv_df_total(path, col_names):
    df = pd.read_csv(path, header=None, skiprows=[0], names=col_names)
    return df


if __name__ == "__main__":
    main()
