from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Activation, AveragePooling3D
import tensorflow as tf


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
    else:
        pool_size1 = [1, 1, 1]
        pool_size2 = [1, 1, 1]
        pool_size3 = [1, 1, 1]

    # [5, 10, 20]
    if kernel_time <= 20:
        kernel_size2 = [int(kernel_time / 2), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 4), kernel_xy, kernel_xy]
    # [40, 60, 80]
    elif kernel_time <= 80:
        kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 16), kernel_xy, kernel_xy]
    # [100]
    else:
        kernel_size2 = [int(kernel_time / 4), kernel_xy, kernel_xy]
        kernel_size3 = [int(kernel_time / 20), kernel_xy, kernel_xy]

    kernel_size4 = [1, kernel_xy, kernel_xy]

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


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,
                                            max_val=1.0, filter_size=11,
                                            filter_sigma=1.5, k1=0.01, k2=0.03))
