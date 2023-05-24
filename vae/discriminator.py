import tensorflow as tf
from keras import Model
from keras.layers import *
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization

class Discriminator:
    @staticmethod
    def discriminator_loss(real_disc_out, fake_disc_out):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        disc_loss_total = 0.0
        for i in range(len(real_disc_out)):
            disc_real_loss = cross_entropy(tf.ones_like(real_disc_out[i]), real_disc_out[i])
            disc_fake_loss = cross_entropy(tf.zeros_like(fake_disc_out[i]), fake_disc_out[i])
            disc_loss = disc_real_loss + disc_fake_loss
            disc_loss_total += disc_loss
        disc_loss_total /= 3
        return disc_loss_total

    @staticmethod
    def generator_loss(fake_disc_out):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gen_loss_total = 0.0
        for i in range(len(fake_disc_out)):
            gen_loss = cross_entropy(tf.ones_like(fake_disc_out[i]), fake_disc_out[i])
            gen_loss_total += gen_loss
        gen_loss_total /= 3
        return gen_loss_total

    @staticmethod
    def scale_discriminator(image_size):
        # Input layer
        image = Input(shape=(image_size, image_size, 3))

        # Scale 1: 256x256
        scale_1 = Conv2D(32, (4, 4), strides=(2, 2), padding='same')(image)
        scale_1 = LeakyReLU(alpha=0.2)(scale_1)

        # Scale 2: 128x128
        scale_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(scale_1)
        scale_2 = InstanceNormalization()(scale_2)
        scale_2 = LeakyReLU(alpha=0.2)(scale_2)

        # Scale 3: 64x64
        scale_3 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(scale_2)
        scale_3 = InstanceNormalization()(scale_3)
        scale_3 = LeakyReLU(alpha=0.2)(scale_3)

        # Scale 4: 32x32
        scale_4 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(scale_3)
        scale_4 = InstanceNormalization()(scale_4)
        scale_4 = LeakyReLU(alpha=0.2)(scale_4)

        # Output layer
        output = Conv2D(1, (4, 4), padding='same')(scale_4)

        return Model(inputs=image, outputs=output)

    @staticmethod
    def build_discriminator(image_size):
        size_h = image_size // 2
        size_q = image_size // 4
        outputs = []
        # Input layer
        image = Input(shape=(image_size, image_size, 3))
        image_h = tf.image.resize(image, [size_h,size_h])
        image_q = tf.image.resize(image, [size_q,size_q])

        disc = Discriminator.scale_discriminator(image_size)(image)
        outputs.append(disc)
        disc_h = Discriminator.scale_discriminator(size_h)(image_h)
        outputs.append(disc_h)
        disc_q = Discriminator.scale_discriminator(size_q)(image_q)
        outputs.append(disc_q)
        return Model(inputs=image, outputs=outputs)