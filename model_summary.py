from tensorflow.keras.models import load_model
from func_3Dver import load_data_predict, split_num, img_show, img_compare, pre_concatenate
import numpy as np
import os
import pickle
from global_value import EPOCHS, BATCH_SIZE, dataset_num, date, model_name


def main():
    result_dirpath = date + "\\" + model_name + "\\"   # k_20
    dirname = result_dirpath + 'model'  # k_20
    model_path = dirname + f'\model_dataset={dataset_num}_e={EPOCHS}_b={BATCH_SIZE}.h5'
    model = load_model(model_path)
    model.summary()




if __name__ == "__main__":
    main()
