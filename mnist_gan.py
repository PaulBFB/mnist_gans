import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from numpy.random import randn, randint
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.datasets.mnist import load_data


def generate_true_sample(data: np.array,
                         number_samples: int) -> (np.ndarray, np.ndarray):
    # random point
    random_index = randint(0, data.shape[0], number_samples)
    image = data[random_index]
    # set lables to 1 (real)
    label = np.ones((number_samples, 1))
    return image, label


def plot_generated_images(images,
                          epoch: int,
                          n: int = 10):
    for i in range(n ** 2):
        # subplot
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = f'generated_images_epoch{epoch}.png'
    plt.savefig(filename)
    plt.close()


class GenerativeAdversarialNetwork:
    def __init__(self,
                 image_dimensions=(28, 28, 1),
                 latent_dim: int = 100):

        self.latent_dim = latent_dim
        # create discriminator model
        discriminator = Sequential()
        discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=image_dimensions))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.4))
        discriminator.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dropout(0.4))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid'))
        # compile discriminator
        opt = Adam(lr=0.0002, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.discriminator_ = discriminator

        # create generator model
        generator = Sequential()
        n_nodes = 128 * 7 * 7
        generator.add(Dense(n_nodes, input_dim=latent_dim))
        # using .2 bcs this is best practice for GANs
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Reshape((7, 7, 128)))
        # upsample * 2
        generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        generator.add(LeakyReLU(alpha=0.2))
        # upsample * 2
        generator.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        self.generator_ = generator

        # "stack" generator & discriminator in a sequential model
        # freeze the discriminator in order to force the generator to converge
        self.discriminator_.trainable = False
        gan = Sequential()
        gan.add(self.generator_)
        gan.add(self.discriminator_)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        gan.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan_ = gan

        # load mnist + prepare the data
        (x, _), (_, _) = load_data()
        # add color channel
        x = np.expand_dims(x, axis=-1)
        # convert to floats
        x = x.astype('float32')
        # scale color to [0,1]
        x = x / 255.0
        self.mnist_x_ = x

    def plot_discriminator(self):
        plot_model(self.discriminator_,
                   to_file='./img/discriminator_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def plot_generator(self):
        plot_model(self.generator_,
                   to_file='./img/generator_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def plot_gan(self):
        plot_model(self.gan_,
                   to_file='./img/GAN_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def show_generator(self):
        self.generator_.summary()

    def show_discriminator(self):
        self.discriminator_.summary()

    def show_gan(self):
        self.gan_.summary()

    # generate points in latent space as generator input
    def generate_latent_points(self,
                               number_samples: int) -> np.ndarray:

        x_input = randn(self.latent_dim * number_samples)
        # reshape for the network
        x_input = x_input.reshape(number_samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self,
                              number_samples) -> (np.ndarray, np.ndarray):

        x_input = self.generate_latent_points(number_samples)
        # predict outputs
        x = self.generator_.predict(x_input)
        # set labels to 0 (fake)
        y = np.zeros((number_samples, 1))
        return x, y


if __name__ == '__main__':
    test = GenerativeAdversarialNetwork()
#    test.plot_gan()
#    test.plot_discriminator()
#    test.plot_generator()
#    test.show_generator()
#    test.show_discriminator()
#    test.show_gan()
#    print(test.generate_latent_points(1))
#    print(test.generate_fake_samples(1))
