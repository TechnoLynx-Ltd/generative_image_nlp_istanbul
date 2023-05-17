import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import Model
from keras.layers import *
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization
import os
from tqdm import tqdm

DATA_FOLDER = "celeba_256_1000"
IMAGE_SIZE = 256
LATENT_DIM = 256
LEARNING_RATE = 0.00005
BATCH_SIZE = 8
EPOCHS = 10


def encoder_model():
    image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = SpectralNormalization(Conv2D(32, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(image)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, strides=2, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    x = InstanceNormalization()(x)
    x = Flatten()(x)
    mean = Dense(LATENT_DIM, kernel_initializer='he_normal')(x)
    logvar = Dense(LATENT_DIM, kernel_initializer='he_normal')(x)
    return Model(inputs=image, outputs=[mean, logvar])


def decoder_model():
    # affine transforms false
    # latent dense to channel size, mul by scale add offset
    # dense and wider conv at beginning

    latent = Input(shape=LATENT_DIM)
    x = SpectralNormalization(Dense(4 * 4 * 128, activation='leaky_relu', kernel_initializer='he_normal'))(latent)
    x = Reshape((4, 4, 128))(x)
    x = InstanceNormalization()(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    level1 = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    residual = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)
    level1_up = UpSampling2D(2, interpolation='bicubic')(level1)
    level2 = residual + level1_up

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    residual = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)
    level2_up = UpSampling2D(2, interpolation='bicubic')(level2)
    level3 = residual + level2_up

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    x = SpectralNormalization(Conv2D(128, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    residual = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)
    level3_up = UpSampling2D(2, interpolation='bicubic')(level3)
    level4 = residual + level3_up

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    x = SpectralNormalization(Conv2D(64, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    residual = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)
    level4_up = UpSampling2D(2, interpolation='bicubic')(level4)
    level5 = residual + level4_up

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    x = SpectralNormalization(Conv2D(32, kernel_size=3, activation='leaky_relu', kernel_initializer='he_normal', padding='same'))(x)
    residual = Conv2D(3, kernel_size=1, kernel_initializer='he_normal')(x)
    level5_up = UpSampling2D(2, interpolation='bicubic')(level5)
    level6 = residual + level5_up

    x = InstanceNormalization()(x)
    x = UpSampling2D(2)(x)
    residual = Conv2D(3, kernel_size=3, kernel_initializer='he_normal', padding='same')(x)
    level6_up = UpSampling2D(2, interpolation='bicubic')(level6)
    recon = residual + level6_up
    return Model(inputs=latent, outputs=[recon, level6, level5, level4, level3, level2, level1])


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
        out.append(cv2.imread(os.path.join(DATA_FOLDER, file)))
    return np.array(out)


def calc_metric(dataset_files):
    metric = tf.keras.metrics.MeanSquaredError()
    metric.reset_state()
    dataset_files = np.random.choice(dataset_files, size=1000, replace=False)
    dataset_files_batched = batch_array(dataset_files)
    for files in dataset_files_batched:
        images = load_batch(files)
        images = images.astype(np.float32) / 255
        images = np.clip(images, -1, 1)
        mean, _ = encoder(images)
        metric.update_state(images, decoder(mean)[0])
    return metric.result().numpy()


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    std = tf.exp(logvar * 0.5)
    return eps * std + mean


def loss_fn(x, recons, mean, logvar):
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse = 0
    for level in recons:
        mse += mse_loss(level, x)
        x = tf.image.resize(x, (x.shape[1] // 2, x.shape[2] // 2), method='bicubic')

    mse /= len(recons)

    kld = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar))
    kld /= BATCH_SIZE * IMAGE_SIZE * IMAGE_SIZE
    return mse + kld


def train_for_one_batch(batch):
    batch = batch.astype(np.float32) / 255

    with tf.GradientTape() as tape_encoder, tf.GradientTape() as tape_decoder:
        mean, logvar = encoder(batch)
        latent = reparameterize(mean, logvar)
        recons = decoder(latent)
        loss_value = loss_fn(batch, recons, mean, logvar)
    gradients = tape_decoder.gradient(loss_value, decoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
    gradients = tape_encoder.gradient(loss_value, encoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))

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
    plt.plot(mse)
    plt.show()


def test():
    encoder = keras.models.load_model("encoder.hd5")
    encoder.summary()
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()
    data_files = np.array(os.listdir(DATA_FOLDER))
    np.random.shuffle(data_files)
    for file in tqdm(data_files):
        image = cv2.imread(os.path.join(DATA_FOLDER, file))
        cv2.imshow("original", image)
        image = image.astype(np.float32) / 255
        mean, _ = encoder(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        recon = decoder(mean)[0].numpy()
        recon = (recon.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).clip(0, 255).astype(np.uint8)
        cv2.imshow("reconstruction", recon)
        cv2.waitKey(0)


def test_random_latent():
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()
    while True:
        latent = np.random.random((1, LATENT_DIM))
        image = decoder(latent)[0].numpy()
        image = (image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).clip(0, 255).astype(np.uint8)
        cv2.imshow("generated image", image)
        cv2.waitKey(0)


# encoder = encoder_model()
# encoder.summary()
# decoder = decoder_model()
# decoder.summary()

encoder = keras.models.load_model("encoder.hd5")
encoder.summary()
decoder = keras.models.load_model("decoder.hd5")
decoder.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# train()
test()
# test_random_latent()
