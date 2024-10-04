from PIL import Image
import os
from natsort import natsorted
time_size = 50
xy_size = 3

def main():
    date = "20240110"
    image_num = "img0"
    # JPGファイルがあるディレクトリパス
    # input_folder = r'C:\Users\AIlab\labo\3DCNN\results\\' + date + rf'\result_kernel_{time_size}_{xy_size}_{xy_size}\predict\train_predict_3D\\' + image_num
    input_folder = r"F:\train_data\20240108\0to1199\img0\image\img1"
    # GIFを保存するファイル名とパス
    # output_gif = r'C:\Users\AIlab\labo\3DCNN\results\\' + date + rf'\result_kernel_{time_size}_{xy_size}_{xy_size}\predict/output_{image_num}.gif'
    output_gif = r"F:\train_data\20240108\0to1199\img0\img1_output.gif"
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
