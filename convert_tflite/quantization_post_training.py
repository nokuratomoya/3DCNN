# https://www.tensorflow.org/lite/performance/post_training_quantization 2024/01/30
# Optimazation Methods
# Full interger quantization
import numpy as np
import sys
import cv2
import tensorflow as tf
import glob
from global_value import *
from natsort import natsorted
import os
from tflite_func import load_dataset_predict

# # settings
# input_model = "saved_model_resnet50"
#
# output_model = "./models/output_quant.tflite"
model_save_path = r"C:\Users\AIlab\labo\3DCNN\results\\" + date + "\\result_kernel_" + str(time_size) + "_" + str(
    xy_size) + "_" + str(xy_size) + r"\model\\"
output_model_path = model_save_path + "quantization_out.tflite"
model_name = "model_dataset=" + str(dataset_num) + "_e=" + str(epochs) + "_b=" + str(batch_size)

# load validation set

def main():
    validation_data_set = []
    all_filename = natsorted(os.listdir(dirname_main))
    filename = all_filename[:dataset_total]
    for name in filename:
        x_pre = load_dataset_predict(name)
        x_pre = np.array(x_pre)
        x_pre = x_pre[:, :, :, :, np.newaxis]
        x_pre = x_pre.astype(np.float32)
        validation_data_set.append(x_pre)
    print(validation_data_set[0].dtype)
    print(len(validation_data_set))

    """    for file_name in img_path:
        img = cv2.imread(file_name)  # BGR
        img = cv2.resize(img, (224, 224))
        ary = np.asarray(img, dtype=np.float32)
        ary = np.expand_dims(ary, axis=0)
        mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
        ary = ary - mean
        ary = np.minimum(ary, 127)
        ary = np.maximum(ary, -128)
        validation_data_set.append(ary)"""

    def representative_dataset_gen():
        for i in range(len(validation_data_set)):
            yield [validation_data_set[i]]

    converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path + model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    print("convert done")

    with open(output_model_path, 'wb') as o_:
        o_.write(tflite_quant_model)
    print("write done")


# # quantize
# def representative_dataset_gen(validation_data_set):
#     for i in range(len(validation_data_set)):
#         yield [validation_data_set[i]]


if __name__ == "__main__":
    main()
