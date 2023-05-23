import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import Model
from keras.layers import *
from keras.applications.vgg19 import VGG19
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization
import os
from tqdm import tqdm

DATA_FOLDER = "../../archive/mini"
IMAGE_SIZE = 256
LATENT_DIM = 256
LEARNING_RATE = 0.00001
BATCH_SIZE = 8
EPOCHS = 300


VGG_LOSS_MULTIPLIER = 1
RECON_LOSS_MULTIPLIER = 0.5
KLD_LOSS_MULTIPLIER = 0.0001
DISC_GEN_LOSS_MULTIPLIER = 0.0001


def encoder_model():
    image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = SpectralNormalization(Conv2D(32, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(image)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = Flatten()(x)
    mean = SpectralNormalization(Dense(LATENT_DIM, kernel_initializer='he_normal'))(x)
    logvar = SpectralNormalization(Dense(LATENT_DIM, kernel_initializer='he_normal'))(x)
    return Model(inputs=image, outputs=[mean, logvar])


def build_scale_discriminator():
    # Input layer
    image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Scale 1: 256x256
    scale_1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(image)
    scale_1 = LeakyReLU(alpha=0.2)(scale_1)

    # Scale 2: 128x128
    scale_2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(scale_1)
    scale_2 = InstanceNormalization()(scale_2)
    scale_2 = LeakyReLU(alpha=0.2)(scale_2)

    # Scale 3: 64x64
    scale_3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(scale_2)
    scale_3 = InstanceNormalization()(scale_3)
    scale_3 = LeakyReLU(alpha=0.2)(scale_3)

    # Scale 4: 32x32
    scale_4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(scale_3)
    scale_4 = InstanceNormalization()(scale_4)
    scale_4 = LeakyReLU(alpha=0.2)(scale_4)

    # Output layer
    output = Conv2D(1, (4, 4), padding='same')(scale_4)

    return Model(inputs=image, outputs=output)



def decoder_block(res, channels, lv, norm):
    interpolation = 'nearest' if res.shape[1] < IMAGE_SIZE // 4 else 'bicubic'
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

    img = Conv2D(3, kernel_size=1, activation='tanh', kernel_initializer='glorot_normal')(res)
    return res, img


def decoder_model():
    outputs = []

    lv = Input(shape=LATENT_DIM)
    res = SpectralNormalization(Dense(4 * 4 * 128, activation='leaky_relu', kernel_initializer='he_normal'))(lv)
    res = Reshape((4, 4, 128))(res)
    res = SpectralNormalization(Conv2D(128, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(res)

    img = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(res)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, False)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, False)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, True)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, True)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, True)
    outputs.append(img)

    res, img = decoder_block(res, 128, lv, True)
    outputs.append(img)

    return Model(inputs=lv, outputs=outputs)


def batch_array(array):
    out = []
    for i in range(array.shape[0] // BATCH_SIZE):
        out.append(array[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
    if array.shape[0] % BATCH_SIZE != 0:
        out.append(array[-(array.shape[0] % BATCH_SIZE):])
    return out


def load_batch(file_batch):
    out = []
    for file in file_batch:
        out.append(cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, file)), (IMAGE_SIZE, IMAGE_SIZE)))
    return np.array(out)


def calc_metric(dataset_files):
    metric = tf.keras.metrics.MeanSquaredError()
    metric.reset_state()
    # dataset_files = np.random.choice(dataset_files, size=1000, replace=False)
    dataset_files_batched = batch_array(dataset_files)
    for files in dataset_files_batched:
        images = load_batch(files)
        images = images.astype(np.float32) / 128 - 1
        mean, _ = encoder(images)
        metric.update_state(images, decoder(mean)[-1])
    return metric.result().numpy()


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    std = tf.exp(logvar * 0.5)
    return eps * std + mean



def discriminator_loss(real_disc_out, fake_disc_out):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_real_loss = cross_entropy(tf.ones_like(real_disc_out), real_disc_out)
    disc_fake_loss = cross_entropy(tf.zeros_like(fake_disc_out), fake_disc_out)
    disc_total_loss = disc_real_loss + disc_fake_loss
    return disc_total_loss

def loss_fn(x, recons, mean, logvar, fake_disc_out):
    # discriminator loss
    disc_labels = tf.ones_like(fake_disc_out)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_gen_loss = cross_entropy(disc_labels, fake_disc_out)

    l1_loss = tf.keras.losses.MeanAbsoluteError()
    recon_loss = 0
    x_copy = tf.identity(x)
    for level in recons[::-1]:
        recon_loss += l1_loss(level, x_copy)
        x_copy = tf.image.resize(x_copy, (x_copy.shape[1] // 2, x_copy.shape[2] // 2), method='bicubic')
    recon_loss /= len(recons)

    kld_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar))
    kld_loss /= BATCH_SIZE

    vgg_loss = 0
    vgg_x = vgg19(x)
    vgg_recon = vgg19(recons[-1])
    for i in range(len(vgg_x)):
        vgg_loss += l1_loss(vgg_x[i], vgg_recon[i])
    vgg_loss /= len(vgg_x)

    loss = RECON_LOSS_MULTIPLIER * recon_loss + VGG_LOSS_MULTIPLIER * vgg_loss + KLD_LOSS_MULTIPLIER * kld_loss + disc_gen_loss * DISC_GEN_LOSS_MULTIPLIER
    return loss


def train_for_one_batch(batch):
    batch = batch.astype(np.float32) / 128 - 1

    with tf.GradientTape() as tape_encoder, tf.GradientTape() as tape_decoder, tf.GradientTape() as disc_tape:
        mean, logvar = encoder(batch)
        latent = reparameterize(mean, logvar)
        recons = decoder(latent)
        gens = recons[-1]
        real_disc_out = discriminator(batch, training=True)
        fake_disc_out = discriminator(gens, training=True)

        disc_loss = discriminator_loss(real_disc_out, fake_disc_out)

        loss_value = loss_fn(batch, recons, mean, logvar, fake_disc_out)
    gradients = tape_decoder.gradient(loss_value, decoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
    gradients = tape_encoder.gradient(loss_value, encoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    # with tf.GradientTape() as tape:
    #     latent2 = encoder(recon)
    #     loss_value = loss_fn(latent, latent2)
    # gradients = tape.gradient(loss_value, encoder.trainable_weights)
    # optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))


def train():
    train_data_files = np.array(os.listdir(DATA_FOLDER))
    mse = []
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        np.random.shuffle(train_data_files)
        train_data_files_batched = batch_array(train_data_files)
        for files in tqdm(train_data_files_batched):
            images = load_batch(files)
            train_for_one_batch(images)
        mse.append(calc_metric(train_data_files))
        encoder.save("encoder.hd5")
        decoder.save("decoder.hd5")
        print(f"MSE = {mse[-1]}")

    # encoder.save("encoder.hd5")
    # decoder.save("decoder.hd5")
    plt.plot(mse)
    plt.savefig('loss.png')


def test():
    encoder = keras.models.load_model("encoder.hd5")
    encoder.summary()
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()
    data_files = np.array(os.listdir(DATA_FOLDER))
    np.random.shuffle(data_files)
    for file in tqdm(data_files):
        image = cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, file)), (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imshow("original", image)
        image = image.astype(np.float32) / 128 - 1
        mean, _ = encoder(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        recon = decoder(mean)[-1].numpy()
        recon = ((recon.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
        cv2.imshow("reconstruction", recon)
        cv2.waitKey(0)


def test_random_latent():
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()
    while True:
        latent = np.random.random((1, LATENT_DIM))
        image = decoder(latent)[-1].numpy()
        image = ((image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
        cv2.imshow("generated image", image)
        cv2.waitKey(0)
#
# def discriminator_loss(real_output, fake_output):
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)


encoder = encoder_model()
encoder.summary()
decoder = decoder_model()
decoder.summary()
discriminator = build_scale_discriminator()
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer_disc = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE / 2)




# encoder = keras.models.load_model("encoder.hd5")
# encoder.summary()
# decoder = keras.models.load_model("decoder.hd5")
# decoder.summary()

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
vgg19_selectedLayers = [1, 2, 9, 10, 17, 18]
vgg19_selectedOutputs = [vgg19.layers[i].output for i in vgg19_selectedLayers]
vgg19 = Model(vgg19.inputs, vgg19_selectedOutputs)

train()
# test()
# test_random_latent()
