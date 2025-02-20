import calcMI_main
import NCC_main
from SSIM import calc_any_SSIM, calc_SSIM, calc_3D_SSIM
from NCC_func import calc_any_NCC, calc_3D_NCC
from global_value import model_date, results_date, time_size, xy_size, results_predict_path, eval_file_path, output3D, dataset_total
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

    # # 推論した学習済みモデル
    # model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{dataset_total}"
    # # model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"
    # print(f"{model_path=}")
    # print(f"{results_predict_path=}")
    # print(f"{eval_file_path=}")
    #
    # if output3D:
    #     calc_3D_SSIM(model_path, results_predict_path, eval_file_path)
    #     calc_3D_NCC(model_path, results_predict_path, eval_file_path)
    #
    # elif not output3D:
    #     calc_any_NCC(model_path, results_predict_path, eval_file_path)
    #     calc_any_SSIM(model_path, results_predict_path, eval_file_path)




    #########
    # loop
    spatial_filter_gain_loop = [1, 0.75, 0.5]
    temporal_filter_par_loop = [1, 2, 4, 8]
    loop_list = list(itertools.product(spatial_filter_gain_loop, temporal_filter_par_loop))
    a_list = [0.005375, 0.01075, 0.0215, 0.043, 0.086, 0.172, 0.344]
    filter_std_list = [3.8, 1.9, 7.6]
    # gain = spatial_filter_gain_loop[0]
    # par = temporal_filter_par_loop[0]

    # datanumついているとき
    model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}_datanum{dataset_total}"
    # datanumついてないとき
    # model_path = model_date + "\\" + f"result_kernel_{time_size}_{xy_size}_{xy_size}"

    # a_compare
    for a in a_list:
        results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\20241217//a_compare//a={a}\\predict\\"
        eval_file_path = rf"E:\LNImodel\train_data\20241217//a={a}\gain24_dt2.5\0to99"
        calc_3D_SSIM(model_path, results_predict_path, eval_file_path)
        # calc_3D_NCC(model_path, results_predict_path, eval_file_path)


    # # filter_std compare
    # for filter_std in filter_std_list:
    #     results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\20250117//filter_std_compare//gain{gain}_par{par}_std{filter_std}\\predict\\"
    #     eval_file_path = rf"E:\LNImodel\train_data\20250117//filter_std_compare//gain{gain}_par{par}_std{filter_std}\gain24_dt2.5\0to99"
    #     calc_3D_SSIM(model_path, results_predict_path, eval_file_path)


    # gain, par compare
    # for gain, par in loop_list:
    #     results_predict_path = rf"C:\Users\AIlab\labo\LNImodel\results\20241129\spatiotemporal_compare\gain{gain}_par{par}\\predict\\"
    #     eval_file_path = rf"E:\LNImodel\train_data\20241129\spatiotemporal_compare\gain{gain}_par{par}\gain24_dt2.5\0to99"
    #     calc_3D_SSIM(model_path, results_predict_path, eval_file_path)


if __name__ == "__main__":
    main()
