import keras.models
from keras.layers import *
from keras import Model
from tensorflow_addons.layers import SpectralNormalization
import tensorflow as tf


def encoder_model(latent_dim):
    image = Input(shape=(256, 256, 3))
    x = SpectralNormalization(Conv2D(32, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(image)
    x = LayerNormalization()(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = LayerNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = LayerNormalization()(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = Flatten()(x)
    mean = SpectralNormalization(Dense(latent_dim, kernel_initializer='he_normal'))(x)
    logvar = SpectralNormalization(Dense(latent_dim, kernel_initializer='he_normal'))(x)
    return Model(inputs=image, outputs=[mean, logvar])


def decoder_block(res, channels, lv, interpolation, norm):
    res = UpSampling2D(2, interpolation=interpolation)(res)
    tf.image.resize(res, (res.shape[1] * 2, res.shape[2] * 2), tf.image.ResizeMethod.BICUBIC)
    res = (Conv2D(channels, kernel_size=3, kernel_initializer='he_normal', padding='same'))(res)
    res = LeakyReLU()(res)

    if norm:
        res = LayerNormalization(center=False, scale=False)(res)

        common_dense = Dense(channels * 4, kernel_initializer='he_normal')(lv)
        common_dense = LeakyReLU()(common_dense)
        common_dense = LayerNormalization()(common_dense)
        beta = Dense(channels, kernel_initializer='he_normal')(common_dense)
        beta = Reshape((1, 1, channels))(beta)
        gamma = Dense(channels, kernel_initializer='he_normal')(common_dense)
        gamma = Reshape((1, 1, channels))(gamma)
        res = res * (1 + gamma) + beta

    img = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(res)
    return res, img


def decoder_model(latent_dim):
    outputs = []

    lv = Input(shape=latent_dim)
    res = Dense(4 * 4 * 256, kernel_initializer='he_normal')(lv)
    res = LeakyReLU()(res)
    res = Reshape((4, 4, 256))(res)
    res = Conv2D(256, kernel_size=3, kernel_initializer='he_normal', padding='same')(res)
    res = LeakyReLU()(res)

    img = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(res)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'bilinear', False)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'bilinear', False)
    outputs.append(img)

    res, img = decoder_block(res, 256, lv, 'bilinear', True)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, 'bilinear', True)
    outputs.append(img)

    res, img = decoder_block(res, 64, lv, 'bilinear', True)
    outputs.append(img)

    res, img = decoder_block(res, 32, lv, 'bilinear', True)
    outputs.append(img)

    return Model(inputs=lv, outputs=outputs)


model = keras.models.load_model("decoder.hd5")
model.summary()

encoder = encoder_model(256)
model2 = decoder_model(256)
for i in range(len(model.layers)):
    if not model.layers[i].get_weights():
        continue
    if model.layers[i].__class__.__name__ == 'SpectralNormalization':
        layer = model.layers[i].layer
    else:
        layer = model.layers[i]
    model2.get_layer(layer.name).set_weights(layer.get_weights())
model2.summary()
model2.save("no_spectral_model")
