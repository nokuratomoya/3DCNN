import calcMI_main
import NCC_main
from SSIM import calc_any_SSIM, calc_SSIM, calc_3D_SSIM
from NCC_func import calc_any_NCC, calc_3D_NCC
from global_value import model_date, results_date, time_size, xy_size, results_predict_path, eval_file_path
import os
from natsort import natsorted


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

    eval_file_all = natsorted(os.listdir(eval_file_path))
    for eval_file in eval_file_all:
        # calc_any_NCC(model_path, results_predict_path, eval_file_path)
        calc_any_SSIM(model_path, results_predict_path, eval_file_path)

    # calc_3D_SSIM(model_path, results_predict_path, eval_file_path)
    # calc_3D_NCC(model_path, results_predict_path, eval_file_path)


if __name__ == "__main__":
    main()
