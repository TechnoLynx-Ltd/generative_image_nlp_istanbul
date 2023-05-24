from keras.layers import *
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization
from keras import Model

# ONLY FOR 256*256 IMAGES


def encoder_model(latent_dim):
    image = Input(shape=(256, 256, 3))
    x = SpectralNormalization(Conv2D(32, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(image)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = Flatten()(x)
    mean = SpectralNormalization(Dense(latent_dim, kernel_initializer='he_normal'))(x)
    logvar = SpectralNormalization(Dense(latent_dim, kernel_initializer='he_normal'))(x)
    return Model(inputs=image, outputs=[mean, logvar])


def decoder_block(res, channels, lv, interpolation, norm):
    res = UpSampling2D(2, interpolation=interpolation)(res)
    res = SpectralNormalization(Conv2D(channels, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(res)

    if norm:
        res = InstanceNormalization(center=False, scale=False)(res)

        common_dense = SpectralNormalization(Dense(channels * 4, activation='leaky_relu', kernel_initializer='he_normal'))(lv)
        common_dense = InstanceNormalization()(common_dense)
        beta = SpectralNormalization(Dense(channels, kernel_initializer='he_normal'))(common_dense)
        beta = Reshape((1, 1, channels))(beta)
        gamma = SpectralNormalization(Dense(channels, kernel_initializer='he_normal'))(common_dense)
        gamma = Reshape((1, 1, channels))(gamma)
        res = res * (1 + gamma) + beta

    img = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(res)
    return res, img


def decoder_model(latent_dim):
    outputs = []

    lv = Input(shape=latent_dim)
    res = SpectralNormalization(Dense(4 * 4 * 256, activation='leaky_relu', kernel_initializer='he_normal'))(lv)
    res = Reshape((4, 4, 256))(res)
    res = SpectralNormalization(Conv2D(256, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(res)

    img = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(res)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'nearest', False)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'nearest', False)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'nearest', True)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, 'nearest', True)
    outputs.append(img)

    res, img = decoder_block(res, 64, lv, 'bicubic', True)
    outputs.append(img)

    res, img = decoder_block(res, 32, lv, 'bicubic', True)
    outputs.append(img)

    return Model(inputs=lv, outputs=outputs)
