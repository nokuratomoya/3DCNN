import numpy as np
from PIL import Image
import os
import pickle
import csv
from natsort import natsorted
from matplotlib import pyplot as plt

def main():
    # カテゴリとそれぞれのデータ
    # 相関係数：train, test, スピアマンの順位相関係数：train, test
    categories = ['train', 'test', 'train', 'test']
    val16 = [0.894879899, 0.691546877, 0.882696607, 0.70541825]
    val30 = [0.93416038, 0.929812114, 0.918826109, 0.917311402]
    plt.figure(figsize=(8, 4))
    # カテゴリの数と棒グラフの幅
    num_categories = len(categories)
    bar_width = 0.2

    # カテゴリのインデックス
    indices = np.arange(num_categories)

    # 1つ目のデータを描画
    plt.bar(indices, val16, bar_width, label='before')

    # 2つ目のデータを描画（幅をずらして表示）
    plt.bar(indices + bar_width, val30, bar_width, label='present')

    # グラフの設定
    # plt.xlabel('カテゴリ')
    # plt.ylabel('値')
    # plt.title('二つのデータセットの比較')
    plt.xticks(indices + bar_width / 2, categories)
    plt.legend()
    plt.ylim(0.65, 0.95)
    # plt.ylim(0, 1)
    # グラフを表示
    # plt.tight_layout()
    # plt.axis("off")
    plt.savefig("NCC_spearman_0_65to0_95.png")
    # plt.savefig("NCC_spearman_0to1.png")
    plt.show()



if __name__ == "__main__":
    main()
