import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.datasets.mnist import load_data


def generate_true_sample(data: np.ndarray,
                         number_samples: int) -> (np.ndarray, np.ndarray):
    """
    generate samples from the true dataset as input to the discriminator
    :param data: dataset to pull from
    :param number_samples: samples to create
    :return: ndarray of samples
    """
    # random point
    random_index = randint(0, data.shape[0], number_samples)
    image = data[random_index]
    # set labels to 1 (real)
    label = np.ones((number_samples, 1))
    return image, label


def plot_generated_images(images,
                          epoch: int,
                          n: int = 10):
    """
    save output of the generator to image folder in square subplots
    :param images: images to be saved
    :param epoch: epoch number, used in filename
    :param n: number of samples to be saved in a grid
    :return: None
    """
    for i in range(n ** 2):
        # subplot
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = f'.img/generated_images_epoch_{epoch}.png'
    plt.savefig(filename)
    plt.close()


class GenerativeAdversarialNetwork:
    def __init__(self,
                 image_dimensions=(28, 28, 1),
                 latent_dim: int = 100):
        """
        create GAN object - initialize & compile generator + discriminator, load mnist data + prepare
        :param image_dimensions: input dimensions of the images
        :param latent_dim: size of the latent space (arbitrary)
        """

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
        """
        save discriminator model architecture to img folder
        :return: None
        """
        plot_model(self.discriminator_,
                   to_file='./img/discriminator_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def plot_generator(self):
        """
        save generator model architecture to img folder
        :return: None
        """
        plot_model(self.generator_,
                   to_file='./img/generator_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def plot_gan(self):
        """
        save GAN model architecture to img folder
        :return:
        """
        plot_model(self.gan_,
                   to_file='./img/GAN_architecture.png',
                   show_shapes=True,
                   show_layer_names=True)

    def show_generator(self):
        """
        show summary of generator model architecture / layers
        :return: None
        """
        self.generator_.summary()

    def show_discriminator(self):
        """
        show summary of discriminator model architecture / layers
        :return: None
        """
        self.discriminator_.summary()

    def show_gan(self):
        """
        show summary of GAN model architecture / layers
        :return: None
        """
        self.gan_.summary()

    # generate points in latent space as generator input
    def generate_latent_points(self,
                               number_samples: int) -> np.ndarray:
        """
        create points in the latent space as input for the generator
        :param number_samples: number of samples to create
        :return: ndarray of inputs
        """

        x_input = randn(self.latent_dim * number_samples)
        # reshape for the network
        x_input = x_input.reshape(number_samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self,
                              number_samples) -> (np.ndarray, np.ndarray):
        """
        create fake samples through the generator
        :param number_samples: amount of samples to generate
        :return: tuple of ndarrays (samples and labels)
        """
        x_input = self.generate_latent_points(number_samples)
        # predict outputs
        x = self.generator_.predict(x_input)
        # set labels to 0 (fake)
        y = np.zeros((number_samples, 1))
        return x, y

    def summarize_performance(self,
                              epoch: int,
                              number_samples=100):
        """
        show current accuracy of the model, save current generator configuration into the models_checkpoints folder
        :param epoch: current epoch, used in filename
        :param number_samples: number of samples to use
        :return: None
        """
        # prepare real samples
        true_x, true_y = generate_true_sample(self.mnist_x_, number_samples)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator_.evaluate(true_x, true_y, verbose=0)
        # prepare fake examples
        fake_x, fake_y = self.generate_fake_samples(number_samples)
        # evaluate discriminator on fake examples
        _, accuracy_fake = self.discriminator_.evaluate(fake_x, fake_y, verbose=0)
        # summarize discriminator performance
        print(f'>Accuracy real: {(acc_real * 100):.2f}, fake: {(accuracy_fake * 100):.2f}')
        # save generated images
        plot_generated_images(fake_x, epoch)
        # save the generator model
        filename = f'./model_checkpoints/generator_model_epoch_{epoch + 1}.h5'
        self.generator_.save(filename)

    def train(self,
              n_epochs: int = 100,
              n_batch: int = 256):
        """
        train the complete model for the set number of epochs (currently 100)
        :param n_epochs: number of epochs
        :param n_batch: batch size
        :return: None
        """
        # set batch size / epoch
        batches_per_epoch = int(self.mnist_x_.shape[0] / n_batch)
        half_batch = int(n_batch / 2)

        # manually enumerate epochs
        for epoch in range(n_epochs):
            # enumerate batches over the training set
            for batch in range(batches_per_epoch):
                # get randomly selected samples
                real_x, real_y = generate_true_sample(self.mnist_x_, half_batch)
                # generate fake samples
                fake_x, fake_y = self.generate_fake_samples(half_batch)
                # create training set for the discriminator from real + fake
                x, y = np.vstack((real_x, fake_x)), np.vstack((real_y, fake_y))
                # update discriminator model weights
                d_loss, _ = self.discriminator_.train_on_batch(x, y)
                # prepare points in latent space as input for the generator
                gan_x = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                gan_y = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_.train_on_batch(gan_x, gan_y)
                # print loss for batch
                print(f'> epoch: {epoch + 1}, batch {batch + 1}/{batches_per_epoch}, loss_disc: {d_loss:.3f}, loss_gen: {g_loss:.3f}')
            # every 10 epochs, show performance + checkpoint
            if (epoch + 1) % 10 == 0:
                self.summarize_performance(epoch)


if __name__ == '__main__':
    test = GenerativeAdversarialNetwork()
    test.plot_gan()
    test.plot_discriminator()
    test.plot_generator()
    test.show_generator()
    test.show_discriminator()
    test.show_gan()
    test.train()
