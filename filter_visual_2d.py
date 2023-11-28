from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EPOCHS = 1000
BATCH_SIZE = 16
dataset_num = 21 * 100
date = "20230117"
pre_file = 'adler_data'


def main():
    # 学習済みmodelの読み込み
    dirname = r'C:\Users\AIlab\labo\複合化用NNプログラム_3Dver\\' + date + r'\results\model'
    model_name = dirname + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    model = load_model(model_name)
    model.summary()

    # 可視化対象レイヤー
    vi_layer = model.get_layer(index=0)

    # レイヤーのフィルタ取得
    target_layer = vi_layer.get_weights()[0]
    filter_num = target_layer.shape[4]
    kernel_size = target_layer.shape[0]

    # 出力
    for i in range(filter_num):
        for j in range(kernel_size):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(int(kernel_size / 6 + 1), 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(target_layer[j, :, :, 0, i], cmap="gray")

        plt.show()


if __name__ == "__main__":
    main()
