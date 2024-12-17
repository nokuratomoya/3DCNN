import calcMI_main
import NCC_main
from SSIM import calc_any_SSIM, calc_SSIM, calc_3D_SSIM
from NCC_func import calc_any_NCC, calc_3D_NCC
from global_value import model_date, results_date, time_size, xy_size, results_predict_path, eval_file_path
import os
from natsort import natsorted
import itertools


def main():
    # calcMI_main.main()
    # print("MI finished")
    # NCC_main.main()
    # print("NCC finished")
    # SSIM.calc_SSIM()
    # print("SSIM finished")

    # 推論した学習済みモデル
    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # # 結果の保存先
    # results_path = r"C:\Users\AIlab\labo\LNPmodel\results\\" + results_date + r"\predict\\"

    # eval_file_all = natsorted(os.listdir(eval_file_path))
    # for eval_file in eval_file_all:
    #     # calc_any_NCC(model_path, results_predict_path, eval_file_path)
    #     calc_any_SSIM(model_path, results_predict_path, eval_file_path)

    calc_3D_SSIM(model_path, results_predict_path, eval_file_path)
    # calc_3D_NCC(model_path, results_predict_path, eval_file_path)


    ##########
    # loop
    # spatial_filter_gain_loop = [1, 0.75, 0.5]
    # temporal_filter_par_loop = [1, 2, 4, 8]
    # loop_list = list(itertools.product(spatial_filter_gain_loop, temporal_filter_par_loop))
    #
    # model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"
    # for gain, par in loop_list:
    #     results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\\20241017\maxGP1\gain{gain}_par{par}\\predict\\"
    #     eval_file_path = rf"E:\LNImodel\train_data\20241017\maxGP1\gain{gain}_par{par}\gain10_dt2.5\0to99"
    #     calc_3D_SSIM(model_path, results_predict_path, eval_file_path)


if __name__ == "__main__":
    main()
