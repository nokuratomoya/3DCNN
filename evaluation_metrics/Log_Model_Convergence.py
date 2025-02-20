import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # データセットから関数フィッティングを行い、任意の増加率を下回る収束点を求めるプログラム
    # ファイル名 time or dataset
    file_name ="dataset"
    y_lim = [0.88,0.93]
    # データセットの読み込み
    if file_name == "dataset":
        load_file_path = r"C:\Users\AIlab\labo\3DCNN\results\datasetsize\dataset_size_SSIM_new.csv"
        label = [5, 10, 15, 20, 40, 80, 100, 200, 400, 1200]
    elif file_name == "time":
        load_file_path = r"C:\Users\AIlab\labo\3DCNN\results\timesize\time_size_SSIM_5ms.csv"
    base_data = csv_file_import(load_file_path)
    # 提示されたデータセット
    # x_data = np.array([10, 20, 40, 320, 480, 640])
    # y_data = np.array([0.722157601, 0.752844575, 0.806300629, 0.827630209, 0.822504305, 0.828429174])
    if file_name == "dataset":
        x_data = base_data["dataset_num"]
    elif file_name == "time":
        x_data = base_data["input_time_size"]

    y_data_name = ["SSIM_train", "SSIM_test"]

    for name in y_data_name:
        y_data = base_data[name]
        # フィッティング
        a, b, _ = fitting_model(x_data, y_data)

        # 高解像度の x データを生成
        x_fine = np.linspace(min(x_data), max(x_data), 1000)

        # 増加率の閾値を設定して収束点を計算
        thresholds = [0.01, 0.001, 0.0001, 0.00005, 0.00001]  # 閾値のリスト
        results = {}

        for threshold in thresholds:
            results[f"増加率 < {threshold}"] = find_convergence(threshold, x_fine, a, b)

        # 結果を表示
        for key, value in results.items():
            print(f"{key}: {value}")

        # フィットモデルをプロット
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label="Observed Data", color="blue", zorder=5)
        # plt.plot(x_fine, log_model(x_fine, a, b), label="Logarithmic Fit", color="red")

        # Highlight convergence points
        for threshold in thresholds:
            result = results[f"増加率 < {threshold}"]
            # if isinstance(result, dict):
                # plt.scatter(result["収束点 (データセット数)"], result["収束時の精度"], label=f"Convergence @ Δ < {threshold}", s=100, zorder=6)

        plt.xlabel(f"{file_name} Size", fontsize=12)
        plt.ylabel(f"{name}", fontsize=12)
        plt.ylim(y_lim[0], y_lim[1])
        plt.title(f"{file_name} Size vs. {name}", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


def csv_file_import(file_path):
    # データセットの読み込み
    # CSVファイルの読み込み
    data = pd.read_csv(file_path)
    # データの内容を確認
    data.head()

    return data


# 対数モデルの定義
def log_model(x, a, b):
    return a * np.log(x) + b


def fitting_model(x_data, y_data):
    # フィッティング
    params, covariance = curve_fit(log_model, x_data, y_data)
    a, b = params
    std_err_a = np.sqrt(np.diag(covariance))[0]  # 傾き a の標準誤差

    return a, b, std_err_a


# 増加率に基づく収束点を探索する関数
def find_convergence(threshold_rate, x_fine, a, b):
    for xi in x_fine:
        rate = a / xi  # 対数モデルの増加率 dy/dx
        if abs(rate) < threshold_rate:
            convergence_x = xi
            convergence_y = log_model(xi, a, b)
            return {
                "収束点 (データセット数)": convergence_x,
                "収束時の精度": convergence_y,
                "増加率": rate,
            }
    return "収束点が見つかりませんでした。"


if __name__ == "__main__":
    main()
