import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
import os
from tqdm import tqdm
from discriminator import Discriminator
from dataloader import DataLoader
from model import *


@tf.function
def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    std = tf.exp(logvar * 0.5)
    return eps * std + mean


@tf.function
def loss_fn(x, recons, mean, logvar):
    loss_dict = {}
    l1_loss = tf.keras.losses.MeanAbsoluteError()
    recon_loss = 0
    x_copy = tf.identity(x)
    for level in recons[::-1]:
        recon_loss += l1_loss(level, x_copy)
        x_copy = tf.image.resize(x_copy, (x_copy.shape[1] // 2, x_copy.shape[2] // 2), method='bicubic')
    recon_loss /= len(recons)
    loss_dict["l1"] = recon_loss

    kld_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mean, 2) - tf.exp(logvar))
    kld_loss /= BATCH_SIZE
    loss_dict["kld"] = kld_loss

    vgg_loss = 0
    vgg_x = vgg19(x)
    vgg_recon = vgg19(recons[-1])
    for i in range(len(vgg_x)):
        vgg_loss += l1_loss(vgg_x[i], vgg_recon[i])
    vgg_loss /= len(vgg_x)
    loss_dict["vgg"] = vgg_loss

    loss = RECON_LOSS_MULTIPLIER * recon_loss + VGG_LOSS_MULTIPLIER * vgg_loss + KLD_LOSS_MULTIPLIER * kld_loss
    return loss, loss_dict


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
    with tf.GradientTape() as tape_encoder, tf.GradientTape() as tape_decoder, tf.GradientTape() as disc_tape:
        mean, logvar = encoder(batch)
        latent = reparameterize(mean, logvar)
        recons = decoder(latent)
        gens = recons[-1]
        loss_value, loss_dict = loss_fn(batch, recons, mean, logvar)

        if USE_DISCRIMINATOR:
            real_disc_out = discriminator(batch, training=True)
            fake_disc_out = discriminator(gens, training=True)

            disc_loss = Discriminator.discriminator_loss(real_disc_out, fake_disc_out)
            loss_dict["disc_loss"] = disc_loss
            gen_loss = Discriminator.generator_loss(fake_disc_out)
            loss_dict["gen_loss"] = gen_loss
            # sacle if needed
            disc_loss_mult = 1.0
            disc_loss *= disc_loss_mult
            loss_value = loss_value + gen_loss * GEN_LOSS_MULTIPLIER

    gradients = tape_decoder.gradient(loss_value, decoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
    gradients = tape_encoder.gradient(loss_value, encoder.trainable_weights)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_weights))
    if USE_DISCRIMINATOR:
        gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer_disc.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
    return loss_dict


def train():
    dataloader = DataLoader(DATA_FOLDER, IMAGE_SIZE, BATCH_SIZE)
    mse = []
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        dataloader.init_dataloader()
        for _ in tqdm(range(dataloader.num_batches)):
            images = dataloader.load_next_batch()
            loss_dict = train_for_one_batch(images)
            if USE_DISCRIMINATOR:
                gen_loss = loss_dict["gen_loss"].numpy()
                disc_loss = loss_dict["disc_loss"].numpy()
            kld_loss = loss_dict["kld"].numpy()
            l1_loss = loss_dict["l1"].numpy()
            vgg_loss = loss_dict["vgg"].numpy()
            if USE_DISCRIMINATOR:
                print(f"gen_loss: {gen_loss}, disc_loss: {disc_loss}, l1: {l1_loss}, kld: {kld_loss}, vgg: {vgg_loss}")
            else:
                print(f"l1: {l1_loss}, kld: {kld_loss}, vgg: {vgg_loss}")
        mse.append(calc_metric(dataloader).numpy())
        log_test_image(os.path.join(args.image_log_path, f"{epoch}_test_img.jpg"))
        if not SAVE_ONLY_AT_END:
            encoder.save("encoder.hd5")
            decoder.save("decoder.hd5")
        if USE_DISCRIMINATOR:
            discriminator.save("discriminator.hd5")
        print(f"MSE = {mse[-1]}")

    if SAVE_ONLY_AT_END:
        encoder.save("encoder.hd5")
        decoder.save("decoder.hd5")
    plt.plot(mse)
    plt.savefig('loss.png')


def test():
    data_files = np.array(os.listdir(DATA_FOLDER))
    np.random.shuffle(data_files)
    file = np.random.choice(data_files, size=1, replace=False)[0]
    image = cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, file)), (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imshow("original", image)
    image = image.astype(np.float32) / 128 - 1
    mean, _ = encoder(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    recon = decoder(mean)[-1].numpy()
    recon = ((recon.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
    cv2.imshow("reconstruction", recon)
    cv2.waitKey(0)


def log_test_image(filename):
    data_files = np.array(os.listdir(DATA_FOLDER))
    np.random.shuffle(data_files)
    file = np.random.choice(data_files, size=1, replace=False)[0]
    image = cv2.resize(cv2.imread(os.path.join(DATA_FOLDER, file)), (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32) / 128 - 1
    mean, _ = encoder(image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    recon = decoder(mean)[-1].numpy()
    recon = ((recon.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
    cv2.imwrite(filename, recon)

def test_random_latent():
    latent = np.random.normal(0, 0.05, (1, LATENT_DIM))
    image = decoder(latent)[-1].numpy()
    image = ((image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3) + 1) * 128).clip(0, 255).astype(np.uint8)
    cv2.imshow("generated image", image)
    cv2.waitKey(0)


def test_interpolation():
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


IMAGE_SIZE = 256
LATENT_DIM = 256

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Run training')
parser.add_argument('--test', action='store_true', help='Test model for an image from the training dataset')
parser.add_argument('--test_random_latent', action='store_true',
                    help='Test model for a randomly generated latent vector')
parser.add_argument('--test_interpolation', action='store_true',
                    help='Test model for two different images from the training dataset, then interpolate between them')
parser.add_argument('--data_path', default='datasets/dataset_5k', type=str, help='Path to the training dataset')
parser.add_argument('--image_log_path', default='./logged_images', type=str, help='Path to the save logged images.')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--calc_metric_num_batches', default=100, type=int,
                    help='Number of batches to evaluate metrics on after each epoch')
parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate for training')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train for')
parser.add_argument('--vgg_loss_mul', default=1, type=float, help='Multiplier of VGG loss during training')
parser.add_argument('--recon_loss_mul', default=1, type=float, help='Multiplier of reconstruction loss during training')
parser.add_argument('--kld_loss_mul', default=0.0001, type=float,
                    help='Multiplier of KL divergence loss during training')
parser.add_argument('--gen_loss_mul', default=1.0, type=float, help='Multiplier of adversarial loss during training')
parser.add_argument('--use_discriminator', action='store_true', help='Use discriminator during training')
parser.add_argument('--restart_training', action='store_true',
                    help='If set, load a blank model and restart the training. If not set, reload the existing model and continue the training.')
parser.add_argument('--save_only_at_end', action='store_true',
                    help='If set, only save the model after the whole training process is finished. If not set, save model after each epoch.')
args = parser.parse_args()

DATA_FOLDER = args.data_path
BATCH_SIZE = args.batch_size
CALC_METRIC_NUM_BATCHES = args.calc_metric_num_batches
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs

VGG_LOSS_MULTIPLIER = args.vgg_loss_mul
RECON_LOSS_MULTIPLIER = args.recon_loss_mul
KLD_LOSS_MULTIPLIER = args.kld_loss_mul
GEN_LOSS_MULTIPLIER = args.gen_loss_mul
USE_DISCRIMINATOR = args.use_discriminator
SAVE_ONLY_AT_END = args.save_only_at_end

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
optimizer_disc = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
metric = tf.keras.metrics.MeanSquaredError()

vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
vgg19_selectedLayers = [1, 2, 9, 10, 17, 18]
vgg19_selectedOutputs = [vgg19.layers[i].output for i in vgg19_selectedLayers]
vgg19 = Model(vgg19.inputs, vgg19_selectedOutputs)

if args.restart_training:
    encoder = encoder_model(LATENT_DIM)
    encoder.compile(optimizer)
    encoder.summary()
    decoder = decoder_model(LATENT_DIM)
    decoder.compile(optimizer)
    decoder.summary()
else:
    encoder = keras.models.load_model("encoder.hd5")
    encoder.summary()
    decoder = keras.models.load_model("decoder.hd5")
    decoder.summary()

if USE_DISCRIMINATOR:
    discriminator = Discriminator.build_discriminator(IMAGE_SIZE)
    discriminator.compile(optimizer_disc)
    discriminator.summary()

if args.train:
    train()
if args.test:
    test()
if args.test_random_latent:
    test_random_latent()
if args.test_interpolation:
    test_interpolation()
