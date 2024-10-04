import numpy as np
from PIL import Image
import os
import pickle
import csv
from natsort import natsorted
# ファイルのimport
from global_value import BATCH_SIZE, dataset_num, date, pixel_size, results_date

stim_head = 201 + 1


def main():
    resize_img()


def resize_img():
    # predict_img_dir = r"C:\Users\AIlab\labo\LNImodel\results\\" + results_date + r"\predict\\"
    # t_v = ["train", "test"]
    # for temp in t_v:
    #     pre_file_path = predict_img_dir + temp + r"_predict"
    #     pre_file_name_all = natsorted(os.listdir(pre_file_path))
    #
    #     os.makedirs(pre_file_path + "\\resize_img", exist_ok=True)
    #
    #     stim_image_path = r"F:\train_data\20240108\0to1199"
    save_resize_img_path = r"F:\train_data\20240108\0to1199_resized"
    os.makedirs(save_resize_img_path + "\\save_npy", exist_ok=True)
    original_img_path = r"F:\train_data\20240108\0to1199"
    img_name_all = natsorted(os.listdir(original_img_path))
    for img_name in img_name_all:
        ori_img_path = os.path.join(original_img_path, str(img_name), "image", "img0", f"img0_{stim_head}.jpg")
        save_path = save_resize_img_path + "\\" + str(img_name) + ".jpg"
        crop_img(save_path, ori_img_path, pixel_size)

        ori_npy_path = os.path.join(original_img_path, str(img_name), "img0", f"img0_{stim_head}.npy")
        save_npy_path = save_resize_img_path + "\\save_npy\\" + str(img_name) + ".npy"
        resize_npy_save(save_npy_path, ori_npy_path)


def crop_img(savefile, path, crop_w_h):
    im = Image.open(path)
    img_width, img_height = im.size
    pil_img = im.crop(((img_width - crop_w_h) // 2,
                       (img_height - crop_w_h) // 2,
                       (img_width + crop_w_h) // 2,
                       (img_height + crop_w_h) // 2))
    pil_img.save(savefile)


def resize_npy_save(save_dir, load_npy, resize_w_h=pixel_size):
    img = np.load(load_npy)
    half_size = int(img.shape[0] / 2)
    img = img[half_size - int(resize_w_h / 2):half_size + int(resize_w_h / 2),
          half_size - int(resize_w_h / 2):half_size + int(resize_w_h / 2)]
    np.save(save_dir, img)



if __name__ == '__main__':
    main()
