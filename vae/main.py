import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
import os
from tqdm import tqdm

from dataloader import DataLoader
from model import *


@tf.function
def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    std = tf.exp(logvar * 0.5)
    return eps * std + mean


@tf.function
def loss_fn(x, recons, mean, logvar):
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

    loss = RECON_LOSS_MULTIPLIER * recon_loss + VGG_LOSS_MULTIPLIER * vgg_loss + KLD_LOSS_MULTIPLIER * kld_loss
    return loss


def calc_metric(dataloader):
    print("Evaluating metrics")
    metric.reset_state()
    batches = dataloader.load_random_batches(min(CALC_METRIC_NUM_BATCHES, dataloader.num_batches))
    for batch in tqdm(batches):
        mean, _ = encoder(batch)
        metric.update_state(batch, decoder(mean)[-1])
    return metric.result()


@tf.function
def train_for_one_batch(batch):
    with tf.GradientTape() as tape_encoder, tf.GradientTape() as tape_decoder:
        mean, logvar = encoder(batch)
        latent = reparameterize(mean, logvar)
        recons = decoder(latent)
        loss_value = loss_fn(batch, recons, mean, logvar)
    gradients = tape_decoder.gradient(loss_value, decoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
    gradients = tape_encoder.gradient(loss_value, encoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))


def train():
    dataloader = DataLoader(DATA_FOLDER, IMAGE_SIZE, BATCH_SIZE)
    mse = []
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        dataloader.init_dataloader()
        for _ in tqdm(range(dataloader.num_batches)):
            images = dataloader.load_next_batch()
            train_for_one_batch(images)
        mse.append(calc_metric(dataloader).numpy())
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


def test_interpolation():
    encoder = keras.models.load_model("encoder.hd5")
    encoder.summary()
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()
    while True:
        data_files = np.array(os.listdir(DATA_FOLDER))
        data_files = np.random.choice(data_files, size=2, replace=False)
        img1 = cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, data_files[0])), (IMAGE_SIZE, IMAGE_SIZE))
        img2 = cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, data_files[1])), (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imshow("original1", img1)
        cv2.imshow("original2", img2)
        img1 = img1.astype(np.float32) / 128 - 1
        img2 = img2.astype(np.float32) / 128 - 1
        mean1, _ = encoder(img1.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        mean2, _ = encoder(img2.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        inter_mean = (mean1 + mean2) / 2
        recon1 = decoder(mean1)[-1].numpy()
        recon2 = decoder(mean2)[-1].numpy()
        inter_recon = decoder(inter_mean)[-1].numpy()
        recon1 = ((recon1.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
        recon2 = ((recon2.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
        inter_recon = ((inter_recon.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
        cv2.imshow("reconstruction1", recon1)
        cv2.imshow("reconstruction2", recon2)
        cv2.imshow("interpolated", inter_recon)
        cv2.waitKey(0)


DATA_FOLDER = "datasets/celeba_256_1000"
IMAGE_SIZE = 256
LATENT_DIM = 256
BATCH_SIZE = 16
CALC_METRIC_NUM_BATCHES = 100
LEARNING_RATE = 0.0001
EPOCHS = 10


VGG_LOSS_MULTIPLIER = 1
RECON_LOSS_MULTIPLIER = 1
KLD_LOSS_MULTIPLIER = 0.0001


optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
metric = tf.keras.metrics.MeanSquaredError()

vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
vgg19_selectedLayers = [1, 2, 9, 10, 17, 18]
vgg19_selectedOutputs = [vgg19.layers[i].output for i in vgg19_selectedLayers]
vgg19 = Model(vgg19.inputs, vgg19_selectedOutputs)


encoder = encoder_model(LATENT_DIM)
encoder.compile(optimizer)
encoder.summary()
decoder = decoder_model(LATENT_DIM)
decoder.compile(optimizer)
decoder.summary()

# encoder = keras.models.load_model("encoder.hd5")
# encoder.summary()
# decoder = keras.models.load_model("decoder.hd5")
# decoder.summary()

# train()
# test()
# test_random_latent()
test_interpolation()
