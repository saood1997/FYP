from __future__ import print_function, division
import datetime
import os
import random
import time
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os, os.path
import tqdm

class DCGAN():
    
    def __init__(self):
        # Shape of generated image - 128x128 RGB
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Shape of the noise vector that will be used as an input for generator
        self.latent_dim = 100
        # Optimizer for discriminator
        optimizer = Adam(lr=0.00002, beta_1=0.5)
        # Optimizer for generator
        optimizer_gen = Adam(lr=0.0002, beta_1=0.5)
        self.discriminator = self.build_discriminator()
        # Load discriminator weights if defined
        if START_EPOCH is not None:
            print("E_D: ", START_EPOCH)
            self.discriminator.load_weights(LOAD_WEIGHTS_PATH + 'faces_d_' + str(START_EPOCH) + '.h5')
        # Binary crossentropy loss function is used on yes/no decisions, e.g., multi-label classification.
        # The loss tells you how wrong your model’s predictions are. For instance, in multi-label problems,
        # where an example can belong to multiple classes at the same time, the model tries to decide
        # for each class whether the example belongs to that class or not.
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        self.generator = self.build_generator()
        # Load generator weights if defined
        if START_EPOCH is not None:
            print("E_G: ", START_EPOCH)
            self.generator.load_weights(LOAD_WEIGHTS_PATH + 'faces_g_' + str(START_EPOCH) + '.h5')
        # Generator input is just noise matrix of size 'batch_size' x 'latent_dim'
        z = Input(shape=(self.latent_dim,))
        # And its output - generated image
        img = self.generator(z)
        self.discriminator.trainable = False
        # Then we pass generated image to discriminator
        # and it returns probability whether or not
        # it is a real training image or a fake image from the generator
        valid = self.discriminator(img)
        # Compose both networks into a single Model
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(8 * 8 * 1024, activation="linear", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=512, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[2, 2],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=self.channels, kernel_size=[5, 5], strides=[1, 1],
                                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                                  padding="same"))
        model.add(Activation("tanh"))
        print("Generator:")
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), input_shape=self.img_shape,
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=128, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=512, kernel_size=[5, 5], strides=[1, 1],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=1024, kernel_size=[5, 5], strides=[2, 2],
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.02),
                         padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.add(Activation("sigmoid"))
        print("Discriminator:")
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def get_batches(self, data, batch_size):
        batches = []
        for i in range(int(data.shape[0] // batch_size)):
            batch = data[i * batch_size:(i + 1) * batch_size]
            augmented_images = []
            for img in batch:
                image = Image.fromarray(img)
                # Flip some images horizontally for better results
                if random.choice([True, False]):
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                augmented_images.append(np.asarray(image))
            batch = np.asarray(augmented_images)
            # Normalize images from 0 / 255 to -1 / 1
            normalized_batch = (batch / 127.5) - 1.0
            batches.append(normalized_batch)
        return batches

    def getTrainingData(self):
        inputList = []
        for file in glob(INPUT_DATA_DIR+'/'+'*'):
            inputList.append(np.asarray(Image.open(file).resize((self.img_rows, self.img_cols))))
        return  np.asarray(inputList)
                                            
    
    def train(self, epochs, batch_size=64):
        x_train = self.getTrainingData()
        print(len(x_train))
        # Add noise to the vector of labels for valid images
        valid = np.ones((batch_size, 1))
        # Generate vector of labels for fake images
        fake = np.zeros((batch_size, 1))
        epoch_n = 0
        d_losses = []
        g_losses = []
        for _ in range(epochs):
            epoch_n += 1
            start_time = time.time()
            mini_epoch_n = 0
            batch = self.get_batches(x_train, batch_size)
            print(len(batch))
            for imgs in batch:
                mini_epoch_n += 1
                # ========================== Main training loop start =======================================
                # For each image in batch generate noise vector
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                # Using those vectors, generate fake images
                # At this point discriminator is not trainable
                gen_imgs = self.generator.predict(noise)
                # Make discriminator trainable
                self.discriminator.trainable = True
                # Train discriminator to distinct fake and real images
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                # Calculate average discriminator loss on both valid and fake images
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # Now make discriminator not trainable
                self.discriminator.trainable = False
                # And train generator
                # Essentially, you tell to trainable part of the model - generator - to correct weights
                # so that generator output (fake images)
                # for current noise vectors will be evaluated by model as the real images
                g_loss = self.combined.train_on_batch(noise, valid)
                # ========================== Main training loop end =======================================
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                print("Batch " + str(mini_epoch_n) + " in epoch " + str(epoch_n) + " with D: " + str(
                    d_loss) + " G: " + str(g_loss) + " finished in " + str(time.time() - start_time))

            print("Epoch " + str(epoch_n) + " finished in " + str(time.time() - start_time))
            # Save generator and discriminator weights as after this epoch
            self.generator.save_weights(SAVE_PATH + 'faces_g_' + str(epoch_n) + '.h5')
            self.discriminator.save_weights(SAVE_PATH + 'faces_d_' + str(epoch_n) + '.h5')
            # Save sample images
            self.save_imgs(epoch_n)
            # Plot losses
            plt.plot(d_losses, label='Discriminator', alpha=0.6)
            plt.plot(g_losses, label='Generator', alpha=0.6)
            plt.title("Losses")
            plt.legend()
            plt.savefig(OUTPUT_DIR + "losses_" + str(epoch_n) + ".png")
            plt.close()
            break

    def show_samples(self, sample_images, name, epoch):
        figure, axes = plt.subplots(1, len(sample_images), figsize=(128, 128))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(name + "faces_" + str(epoch) + "_" + str(index) + ".png")
        plt.close()

    def save_imgs(self, epoch):
        r = 10
        noise = np.random.uniform(-1, 1, (r, self.latent_dim))
        samples = self.generator.predict(noise)
        sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
        self.show_samples(sample_images, OUTPUT_DIR, epoch)


# Path to project folder
BASE_PATH = "/home/raza/Documents/FYP_1"
#BASE_PATH = "/home/raza/Downloads/keras/DeepLearingWorkSpace"
# Path to folder with checkpoints and which epoch to load
START_EPOCH = 1
LOAD_WEIGHTS_PATH = BASE_PATH + '/models/'

OUTPUT_DIR = BASE_PATH + '/generateImages/'
#SAVE_PATH = BASE_PATH + '/models/{date:%Y-%m-%d_%H-%M-%S}/'.format(date=datetime.datetime.now())

dcgan = DCGAN()
dcgan.save_imgs(1)
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
#dcgan.train(epochs=200, batch_size=128)
