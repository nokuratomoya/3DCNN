# NCC(Normalized Cross-Correlation):正規化相互相関の計算

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import math
import pickle
import os
from global_value import time_size, xy_size, E
from NCC_func import NCC


def main():
    # for i, time in enumerate(time_size):
    #     for j, xy in enumerate(xy_size):
    #         model_name = f"result_kernel_{time}_{xy}_{xy}"
    #
    #         NCC(model_name, E[i][j])

    model_name = f"result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}"
    NCC(model_name, E)


if __name__ == "__main__":
    main()
