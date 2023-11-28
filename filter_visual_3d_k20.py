from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

EPOCHS = 1263
BATCH_SIZE = 16
dataset_num = 40 * 116
date = "20230126"


# pre_file = 'adler_data'


def main():
    # 学習済みmodelの読み込み
    dirname = r'C:\Users\AIlab\labo\複合化用NNプログラム_3Dver\\' + date + r'\results_k20\model'
    model_name = dirname + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    model = load_model(model_name)
    model.summary()

    # 保存フォルダ作成
    pre_dirpath = r'C:\Users\AIlab\labo\複合化用NNプログラム_3Dver\\' + date + r'\results_k20\filter_weight\\'
    os.makedirs(pre_dirpath, exist_ok=True)
    save_file = pre_dirpath + f'dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}'
    # 可視化対象レイヤー
    vi_layer = model.get_layer(index=0)

    # レイヤーのフィルタ取得
    target_layer = vi_layer.get_weights()[0]
    filter_num = target_layer.shape[4]
    kernel_size = target_layer.shape[0]

    # フィルタ一つずつプロット
    # plot_fw_one(filter_num, kernel_size, target_layer, save_file)

    # フィルタの平均をプロット
    plot_fw_ave(filter_num, kernel_size, target_layer, save_file)


def plot_fw_one(filter_num, kernel_size, target_layer, save_file):
    for j in range(filter_num):
        filter_weight = []
        for i in range(kernel_size):
            filter_weight.append(target_layer[i, :, :, 0, j])
        filter_weight = np.array(filter_weight)
        # 出力
        w_x, w_y, w_z = np.meshgrid(range(filter_weight.shape[0]), range(filter_weight.shape[1]),
                                    range(filter_weight.shape[2]))
        fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(111, projection='3d', aspect='auto')
        ax = fig.add_subplot(111, projection='3d', xmargin=0.1, ymargin=0.5, zmargin=0.5)
        ax.set_xlabel("time", size=10, color="black")
        ax.set_ylabel("x:horizontal", size=10, color="black")
        ax.set_zlabel("y:vertical", size=10, color="black")

        # 軸、目盛りの消去
        ax.axes.yaxis.set_ticks([])
        ax.axes.zaxis.set_ticks([])
        # ax.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        plt.xticks(np.arange(0, filter_weight.shape[0]))

        # グラフ回転
        ax.view_init(elev=30, azim=-110)

        # y軸の反転（フィルターのx軸を反転させる）
        ax.invert_yaxis()

        # アスペクト比変更
        ax.set_box_aspect((40, 5, 5))

        sc = ax.scatter(w_x, w_y, w_z, s=30, c=filter_weight, alpha=0.5, cmap='bwr', edgecolors='black', marker="s")
        # fig.colorbar(sc)
        # plt.savefig(save_file + f"_filter{j}_w.jpg")  # filter_weight
        plt.show()
        plt.close()


# フィルター16枚の平均
def plot_fw_ave(filter_num, kernel_size, target_layer, save_file):
    """
    for j in range(filter_num):
        filter_weight = []
        for i in range(kernel_size):
            filter_weight.append(target_layer[i, :, :, 0, j])
    """
    # フィルタの値の平均化
    filter_weight = []
    for i in range(kernel_size):
        filter_weight_ave = np.zeros((target_layer.shape[1], target_layer.shape[2]))
        for j in range(filter_num):
            filter_weight_ave += target_layer[i, :, :, 0, j]
        filter_weight.append(filter_weight_ave)
    filter_weight = np.array(filter_weight) / filter_num

    # 出力
    w_x, w_y, w_z = np.meshgrid(range(filter_weight.shape[0]), range(filter_weight.shape[1]),
                                range(filter_weight.shape[2]))
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d', aspect='auto')
    ax = fig.add_subplot(111, projection='3d', xmargin=0.1, ymargin=0.5, zmargin=0.5)
    # ax.set_xlabel("time", size=10, color="black")
    ax.set_ylabel("x:horizontal", size=10, color="black")
    ax.set_zlabel("y:vertical", size=10, color="black")

    # 軸、目盛りの消去
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    # ax.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
    plt.xticks(np.arange(0, filter_weight.shape[0]))

    # グラフ回転
    ax.view_init(elev=30, azim=-110)

    # y軸の反転（フィルターのx軸を反転させる）
    ax.invert_yaxis()

    # アスペクト比変更
    ax.set_box_aspect((40, 5, 5))

    sc = ax.scatter(w_x, w_y, w_z, s=30, c=filter_weight, alpha=0.5, cmap='bwr', edgecolors='black', marker="s")
    # fig.colorbar(sc)
    # plt.savefig(save_file + f"_filter_w_ave_k20.jpg")  # filter_weight
    plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
