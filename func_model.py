import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Activation, AveragePooling3D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, UpSampling2D, Reshape, \
    Activation, TimeDistributed, ConvLSTM2D, BatchNormalization
# from keras.layers.rnn.conv_lstm import ConvLSTM2D
# from tensorflow.keras.layers import Conv2D, LSTM, Dense, Activation, AveragePooling2D, Reshape
import tensorflow as tf
import sys


def model_build(kernel_time, kernel_xy, filter_size, input_shape):
    kernel_size1 = [kernel_time, kernel_xy, kernel_xy]

    if kernel_time == 100:
        pool_size1 = [4, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 50:
        pool_size1 = [2, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 20:
        pool_size1 = [2, 1, 1]
        pool_size2 = [2, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 10:
        pool_size1 = [1, 1, 1]
        pool_size2 = [2, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 40:
        pool_size1 = [2, 1, 1]
        pool_size2 = [4, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 30:
        pool_size1 = [2, 1, 1]
        pool_size2 = [3, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 150:
        pool_size1 = [6, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 200:
        pool_size1 = [8, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 250:
        pool_size1 = [10, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    elif kernel_time == 300:
        pool_size1 = [12, 1, 1]
        pool_size2 = [5, 1, 1]
        pool_size3 = [5, 1, 1]
    else:
        pool_size1 = [1, 1, 1]
        pool_size2 = [1, 1, 1]
        pool_size3 = [1, 1, 1]

    # # [5, 10, 20]
    # if kernel_time == 1:
    #     kernel_size2 = [1, kernel_xy, kernel_xy]
    #     kernel_size3 = [1, kernel_xy, kernel_xy]
    # elif kernel_time <= 20:
    #     kernel_size2 = [int(kernel_time / 2), kernel_xy, kernel_xy]
    #     kernel_size3 = [int(kernel_time / 4), kernel_xy, kernel_xy]
    # # [40, 60, 80]
    # elif kernel_time <= 80:
    #     kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
    #     kernel_size3 = [int(kernel_time / 16), kernel_xy, kernel_xy]
    # # [100]
    # else:
    #     kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
    #     kernel_size3 = [int(kernel_time / 20), kernel_xy, kernel_xy]
    kernel_size2 = [int(kernel_time / pool_size1[0]), kernel_xy, kernel_xy]
    kernel_size3 = [int(kernel_time / (pool_size1[0] * pool_size2[0])), kernel_xy, kernel_xy]
    kernel_size4 = [int(kernel_time / (pool_size1[0] * pool_size2[0] * pool_size3[0])), kernel_xy, kernel_xy]

    if kernel_size4[0] != 1:
        sys.exit("kernel_size4[0] != 1")

    print(kernel_size1)
    model = Sequential(
        layers=[

            Conv3D(filters=filter_size, kernel_size=kernel_size1, kernel_initializer='lecun_uniform', padding='same',
                   input_shape=input_shape
                   ),  # (100, 30, 30, 1) => (100, 30, 30, 1)
            # BatchNormalization(),
            Activation('relu'),
            Conv3D(filters=filter_size, kernel_size=kernel_size1, padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            AveragePooling3D(pool_size=pool_size1),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # MaxPooling3D(pool_size=(4, 1, 1)),  # (100, 30, 30, 1) => (25, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=filter_size * 2, kernel_size=kernel_size2, padding='same'),
            # BatchNormalization(),
            Activation('relu'),
            AveragePooling3D(pool_size=pool_size2),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # MaxPooling3D(pool_size=(5, 1, 1)),  # (25, 30, 30, 1) => (5, 30, 30, 1)
            # Dropout(0.25),

            Conv3D(filters=filter_size * 4, kernel_size=kernel_size3, padding='same'),
            Activation('relu'),
            # (5, 30, 30, 1) => (5, 30, 30, 1)

            AveragePooling3D(pool_size=pool_size3),  # (5, 30, 30, 1) => (1, 30, 30, 1)
            # MaxPooling3D(pool_size=(5, 1, 1)),  # (5, 30, 30, 1) => (1, 30, 30, 1)

            Conv3D(filters=1, kernel_size=kernel_size4, padding='same'),
            # BatchNormalization(),
            Activation('sigmoid')

        ]
    )

    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.compile(loss=ssim_loss, optimizer='adam', metrics=[ssim_loss])
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def model_cnn_LSTM(kernel_time, kernel_xy, filter_size, input_shape):
    """
        # 新しいモデルの定義
        model = Sequential(
            layers=[
                Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same', input_shape=input_shape),
                Activation('relu'),
                TimeDistributed(LSTM(100, return_sequences=True)),
                Activation('relu'),
                Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same'),
                Activation('relu'),
                TimeDistributed(LSTM(100, return_sequences=False)),
                Activation('sigmoid')

            ]
        )

        # 時系列情報を LSTM で処理するためにデータを整形

        #  model.add(Activation("sigmoid"))

        model.compile(loss=ssim_loss, optimizer='adam', metrics=[ssim_loss])
        model.summary()
    """

    # Conv2D レイヤーを使って時空間情報を捉える
    """    model.add(Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1)))"""
    model = Sequential()
    model.add(Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1)))

    # 時系列情報を LSTM で処理するためにデータを整形
    # model.add(Reshape((120, filter_size * 5)))  # (30, 30, filter_size * 5) => (30, filter_size * 5)
    # model.add(Reshape((100, 120, 120, 1)))  # (30, 30, filter_size * 5) => (30, filter_size * 5)
    model.add(Reshape((50, filter_size * 120 * 120)))

    # LSTM レイヤーを使用して時系列情報を処理
    model.add(LSTM(units=32, return_sequences=True))
    model.add(LSTM(units=32, return_sequences=False))

    # 2D 空間に戻す
    model.add(Reshape((120, 120, 32)))  # (32,) => (1, 1, 32)

    # Conv2D レイヤーを再度使用して最終的な特徴を抽出
    model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same'))
    model.add(Activation('sigmoid'))

    # モデルのコンパイル
    model.compile(loss="mse", optimizer='adam', metrics=["loss"])

    # モデルの概要表示
    model.summary()

    return model


def model_ConvLSTM2D(kernel_time, kernel_xy, filter_size, input_shape):
    # input_shape = (100, 120, 120, 1)
    # kernel_time = 100
    # kernel_xy = 3
    # filter_size = 4

    model = Sequential(
        layers=[
            ConvLSTM2D(filters=filter_size, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='relu', input_shape=input_shape),
            ConvLSTM2D(filters=filter_size, kernel_size=(3, 3), padding='same', return_sequences=True,
                       activation='relu'),
            # ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=False, activation='sigmoid')
            ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=False, activation='sigmoid')

        ]
    )

    # 時系列情報を LSTM で処理するためにデータを整形

    #  model.add(Activation("sigmoid"))

    model.compile(loss=ssim_loss, optimizer='adam', metrics=[ssim_loss])
    # model.compile(loss=ssim_loss, optimizer='adam', metrics=[ssim_loss])
    model.summary()

    return model


# 参考文献
# Reconstruction of Natural Visual Scenes from Neural Spikes with Deep Neural Networks

def SID_model_build():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=7, padding="same", strides=(2, 2), activation='relu', input_shape=(64, 64, 1)))
    model.add(Conv2D(128, kernel_size=5, padding="same", strides=(2, 2), activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding="same", strides=(2, 2), activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding="same", strides=(2, 2), activation='relu'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, kernel_size=3, padding="same", strides=(1, 1), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, kernel_size=3, padding="same", strides=(1, 1), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=5, padding="same", strides=(1, 1), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, kernel_size=7, padding="same", strides=(1, 1), activation='relu'))
    model.add(Conv2D(1, kernel_size=1, padding="same", strides=(1, 1), activation='sigmoid'))

    # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.compile(loss=ssim_loss, optimizer='adam', metrics=[ssim_loss])
    model.summary()

    return model


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,
                                            max_val=1.0, filter_size=11,
                                            filter_sigma=1.5, k1=0.01, k2=0.03))

# SID_model_build()
# model_build(50, 3, 4, (50, 120, 120, 1))
# model_ConvLSTM2D(100, 3, 4, (100, 120, 120, 1))

# model_cnn_LSTM(100, 3, 4, )
