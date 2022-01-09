#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:55:55 2020

@author: luofan
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from numpy import zeros
from numpy import ones
from numpy import hstack
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model


data = pd.read_csv('dataset.csv')
data=data['H_S']
data = data.tolist()
data = np.array(data).reshape(-1,2)

def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))  # dense: a hidden layer has 25 nodes
    model.add(Dense(1, activation='sigmoid'))  # output layer has one node for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # minimize the loss
    return model

model = define_discriminator()

def generate_real_samples(n):
    X = data
    y = ones((n, 1))  # class labels
    return X, y

def generate_fake_samples(n):
    X1 = -1 + rand(n) * 2
    X2 = -1 + rand(n) * 2
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    y = zeros((n, 1)) # class labels
    return X, y

def train_discriminator(model, n_epochs=1000, n_batch=1276):  # train 1000 epochs,every batch:half real, half fake
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        X_real, y_real = generate_real_samples(half_batch)
        model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        model.train_on_batch(X_fake, y_fake)
        _, acc_real = model.evaluate(X_real, y_real, verbose=0)  # real data
        _, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)  # fake data
        print(i, acc_real, acc_fake)
        
model = define_discriminator()
train_discriminator(model)

def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))  # use linear function because we want to output real value vector
    return model

model = define_generator(5)


def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

latent_dim = 5  # five-element vector of Gaussian random numbers
model = define_generator(latent_dim)
generate_fake_samples(model, latent_dim, 100)

def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

latent_dim = 5
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)


def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    y = zeros((n, 1))
    return X, y

def summarize_performance(epoch, generator, discriminator, latent_dim, n=638):
    x_real, y_real = generate_real_samples(n)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print(epoch, acc_real, acc_fake)
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    pyplot.show()

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=1276, n_eval=2000):
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)
        if (i+1) % n_epochs == 0:
            summarize_performance(i, g_model, d_model, latent_dim)
            hs =  pd.DataFrame(x_fake)
            hs.to_csv('out.csv', header=None, index=None)

latent_dim = 5
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
train(generator, discriminator, gan_model, latent_dim)
