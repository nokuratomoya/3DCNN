from PIL import Image
import os
from natsort import natsorted
def main():


    # JPGファイルがあるディレクトリパス
    input_folder = r'C:\Users\AIlab\labo\3DCNN\results\20231205\result_kernel_100_3_3\predict\train_predict_3D\img0'

    # GIFを保存するファイル名とパス
    output_gif = r'C:\Users\AIlab\labo\3DCNN\results\20231205\result_kernel_100_3_3\predict/output_img0.gif'

    # 画像ファイルのリストを取得
    image_list = [f for f in natsorted(os.listdir(input_folder)) if f.endswith('.jpg')]

    # 画像をGIFに変換
    images = []
    for img_name in image_list:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path)
        images.append(img)
    print(len(images))
    # GIFとして保存
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=20, loop=0)


if __name__ == "__main__":
    main()
