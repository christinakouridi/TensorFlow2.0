#!/usr/bin/env python
# coding: utf-8
# author: Christina Kouridi

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from IPython import display
from pathlib import Path
from tensorflow.keras import layers, optimizers, metrics, Model



class GAN:
    def __init__(self, hlayers, noise_dim=100, epochs=100, batch_size=64,
                 learning_rate=2e-4, beta_1=0.5, beta_2=0.999, alpha=0.2,
                 smooth=0.1, num_examples=16, create_gif=False):
        """
        Implementation of a vanilla GAN using TensorFlow 2.0.
        The Generator and Discriminator are described by multilayer perceptrons
        Input:
            hlayers - neurons in the hidden layers of the generator and disciminator
            noise_dim - # of neurons in the input layer of the generator
            epochs - # of training iterations
            batch_size - # of training examples in each batch
            learning_rate - to what extent newly acquired info. overrides old info.
            beta_1 - hyperparameter that controls the exponential decay rate of
                     the moving averages of the gradient (adam optimiser)
            beta_2 - hyperparameter that controls the exponential decay rate of
                     the moving averages of the squared gradient (adam optimiser)
            alpha - gradient of mapping function for negative values (leaky relu)
            smooth - one-sided label smoothing factor
            num_examples - # of sample images to display during training and in final gif
            create_gif - if true, a gif of sample images will be generated
        """

        super(GAN, self).__init__()

        # -------- Initialise hyperparameters --------#
        self.hlayers = hlayers
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.smooth = smooth
        self.num_examples = num_examples
        self.create_gif = create_gif

        # -------- Initialise Generator & Optimizer --------#
        self.g = self.generator(training=True)
        self.g_optimizer = optimizers.Adam(lr=self.lr, beta_1=self.beta_1,
                                           beta_2=self.beta_2)

        # -------- Initialise Discriminator & Optimiser --------#
        self.d = self.discriminator(training=True)
        self.d_optimizer = optimizers.Adam(lr=self.lr, beta_1=self.beta_1,
                                           beta_2=self.beta_2)

        self.seed = tf.random.normal([self.num_examples, self.noise_dim])

        # -------- Set-up for storing & generating gif --------#
        if self.create_gif:
            self.image_dir = Path('./GAN_sample_images')  # creates new folder in cd
            if not self.image_dir.is_dir():
                self.image_dir.mkdir()

            self.filenames = []  # stores filenames of sample images if create_gif is enabled

    def generator(self, training=False):
        """
        Creates a model for forward propagation through the generator
        """
        model = tf.keras.Sequential()

        model.add(layers.Dense(units=self.hlayers["g"][0], input_shape=(self.noise_dim,),
                               activation=tf.nn.relu, kernel_initializer='glorot_normal'))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=self.hlayers["g"][1], activation=tf.nn.relu))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=self.hlayers["g"][2], activation=tf.nn.relu))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=784, activation=tf.nn.tanh))
        return model

    def discriminator(self, training=None):
        """
        Creates a model for forward propagation through the discriminator
        """
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28, 1)))

        model.add(layers.Dense(units=self.hlayers["d"][0], kernel_initializer='glorot_normal'))
        model.add(layers.LeakyReLU(self.alpha))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=self.hlayers["d"][1]))
        model.add(layers.LeakyReLU(self.alpha))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=self.hlayers["d"][2]))
        model.add(layers.LeakyReLU(self.alpha))
        model.add(layers.BatchNormalization(trainable=training))

        model.add(layers.Dense(units=1, activation=tf.nn.sigmoid))
        return model

    @staticmethod
    def discriminator_loss(d_real_logits, d_fake_logits):
        """
        Calculates the cross entropy loss of the discirminator for the real and fake images
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        d_real_loss = tf.reduce_mean(cross_entropy(tf.ones_like(d_real_logits), d_real_logits))
        d_fake_loss = tf.reduce_mean(cross_entropy(tf.zeros_like(d_fake_logits), d_fake_logits))

        d_loss = d_real_loss + d_fake_loss
        return d_loss

    @staticmethod
    def generator_loss(g_logits):
        """
        Calculates the cross entropy loss of the generator
        """
        # This method returns a helper function to compute cross entropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        g_loss = tf.reduce_mean(cross_entropy(tf.ones_like(g_logits), g_logits))
        return g_loss

    @tf.function  # This annotation causes the function to be "compiled"
    def train_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.g(noise, training=True)

            d_real_logits = self.d(real_images, training=True)

            d_real_logits = d_real_logits * (1 - self.smooth)
            d_fake_logits = self.d(fake_images, training=True)

            g_loss = self.generator_loss(d_fake_logits)
            d_loss = self.discriminator_loss(d_real_logits, d_fake_logits)

        g_grads = g_tape.gradient(g_loss, self.g.trainable_variables)
        d_grads = d_tape.gradient(d_loss, self.d.trainable_variables)

        self.g_optimizer.apply_gradients(zip(g_grads, self.g.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.d.trainable_variables))
        return g_loss, d_loss

    def sample_and_save_images(self, epoch):
        """
        Generates a grid with sample images from the generator.
        Images are stored in the GAN_sample_images folder in the local directory.
        """
        # Set training to False in order not to train the batchnorm layer during inference
        predictions = self.g(self.seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        predictions = tf.reshape(predictions, [predictions.shape[0], 28, 28, 1])

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        # saves generated images in the GAN_sample_images folder
        if self.create_gif:
            current_epoch_filename = self.image_dir.joinpath(f"GAN_epoch{epoch}.png")
            self.filenames.append(current_epoch_filename)
            plt.savefig(current_epoch_filename)
        plt.show()

    def generate_gif(self):
        """
        Generates a gif from the exported generated images at each training iteration
        """
        images = []
        for filename in self.filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave("GAN.gif", images)

    def train(self, x_train):
        """
        Main method of the GAN class where training takes place
        """
        g_losses = []  # stores the generator loss for each batch and epoch
        d_losses = []  # stores the discriminator loss for each batch and epoch

        # batch and shuffle the training data
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(self.batch_size)

        for epoch in range(self.epochs):
            start = time.time()

            for batch_images in train_dataset:
                g_loss, d_loss = self.train_step(batch_images)

                g_losses.append(g_loss)
                d_losses.append(d_loss)

            # produce images for the GIF as we go
            display.clear_output(wait=True)
            self.sample_and_save_images(epoch + 1)

            print(
                f'Epoch {epoch+1} | G Loss: {g_loss:.1f} | D Loss: {d_loss:.1f} | Time: {(time.time() - start):.1f} sec')

        # generate after the final epoch
        display.clear_output(wait=True)

        # generate gif
        if self.create_gif:
            self.generate_gif()

        self.sample_and_save_images(self.epochs)

        return g_losses, d_losses