from predict_img_func import predict_image
from global_value import time_size, xy_size, E


def main():

    # for i, time in enumerate(time_size):
    #     for j, xy in enumerate(xy_size):
    #         model_name = f"result_kernel_{time}_{xy}_{xy}"
    #
    #         predict_image(model_name, E[i][j])
    #
    # for j, xy in enumerate(xy_size):
    #     model_name = f"result_kernel_{time_size[0]}_{xy}_{xy}"
    #
    #     predict_image(model_name, E[j])
    # model_name = f"result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}"
    model_name = f"result_kernel_{time_size[0]}_{xy_size[0]}_{xy_size[0]}"
    predict_image(model_name, E)


if __name__ == "__main__":
    main()
