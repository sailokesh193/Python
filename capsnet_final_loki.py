# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:48:27 2018

@author: sailokesh.kukkala
"""

import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
from matplotlib import pylab
from pylab import *
import tensorflow as tf
import keras
import keras.backend as K


from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

K.set_image_data_format('channels_last')

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'data')

train = pd.read_csv('C:/Users/sailokesh.kukkala/Downloads/Train_HI6auGp/Train/train.csv')
test = pd.read_csv('C:/Users/sailokesh.kukkala/Downloads/Test_fCbTej3.csv')

img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'train', img_name)

from scipy.misc import imread

img = imread(filepath,flatten = True)

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

temp = []
for img_name in train.filename:
   image_path = os.path.join(data_dir, 'train', img_name)
   img = imread(image_path)
   img = img.astype('float32')
   temp.append(img)
 
train_x = np.stack(temp)

train_x /= 255.0
train_x = train_x.reshape(-1, 3136).astype('float32') #784

train_y = keras.utils.np_utils.to_categorical(train.label.values)


split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

#Multilayer Perception

# define vars
input_num_units = 3136
hidden_num_units = 50
output_num_units = 10

epochs = 15
batch_size = 128

# import keras modules

from keras.models import Sequential
from keras.layers import InputLayer, Convolution2D, MaxPooling2D, Flatten, Dense

# create model
model = Sequential([
 Dense(units=hidden_num_units, input_dim=input_num_units, activation='relu'),
 Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()


trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))


#CNN

# reshape data
train_x_temp = train_x.reshape(-1, 28, 28, 4)
val_x_temp = val_x.reshape(-1, 28, 28, 4)

# define vars
input_shape = (3136,)
input_reshape = (28, 28, 4)


pool_size = (2, 2)

hidden_num_units = 50
output_num_units = 10

batch_size = 128



model = Sequential([
 InputLayer(input_shape=input_reshape),

Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

Convolution2D(25, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

Convolution2D(25, 4, 4, activation='relu'),

Flatten(),

Dense(output_dim=hidden_num_units, activation='relu'),

Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))

model.summary()

# Begin: Training with data augmentation ---------------------------------------------------------------------#
def train_generator(x, y, batch_size, shift_fraction=0.1):
   train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
   height_shift_range=shift_fraction) # shift up to 2 pixel for MNIST
   generator = train_datagen.flow(x, y, batch_size=batch_size)
   while 1:
     x_batch, y_batch = generator.next()
     yield ([x_batch, y_batch])
 
# Training with data augmentation. If shift_fraction=0., also no augmentation.
trained_model2 = model.fit_generator(generator=train_generator(train_x_temp, train_y, 1000, 0.1),
 steps_per_epoch=int(train_y.shape[0] / 1000),
 epochs=epochs,
 validation_data=[val_x_temp, val_y])
# End: Training with data augmentation -----------------------------------------------------------------------#



#Capsule Network

def CapsNet(input_shape, n_class, routings):
   """
   A Capsule Network on MNIST.
   :param input_shape: data shape, 3d, [width, height, channels]
   :param n_class: number of classes
   :param routings: number of routing iterations
   :return: Two Keras Models, the first one used for training, and the second one for evaluation.
   `eval_model` can also be used for training.
   """
   x = layers.Input(shape=input_shape)

   # Layer 1: Just a conventional Conv2D layer
   conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

   # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
   primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

   # Layer 3: Capsule layer. Routing algorithm works here.
   digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
   name='digitcaps')(primarycaps)

   # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
   # If using tensorflow, this will not be necessary. :)
   out_caps = Length(name='capsnet')(digitcaps)

   # Decoder network.
   y = layers.Input(shape=(n_class,))
   masked_by_y = Mask()([digitcaps, y]) # The true label is used to mask the output of capsule layer. For training
   masked = Mask()(digitcaps) # Mask using the capsule with maximal length. For prediction

   # Shared Decoder model in training and prediction
   decoder = models.Sequential(name='decoder')
   decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
   decoder.add(layers.Dense(1024, activation='relu'))
   decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
   decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

   # Models for training and evaluation (prediction)
   train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
   eval_model = models.Model(x, [out_caps, decoder(masked)])

   # manipulate model
   noise = layers.Input(shape=(n_class, 16))
   noised_digitcaps = layers.Add()([digitcaps, noise])
   masked_noised_y = Mask()([noised_digitcaps, y])
   manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
   return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
   """
   Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
   :param y_true: [None, n_classes]
   :param y_pred: [None, num_capsule]
   :return: a scalar loss value.
   """
   L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
   0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

   return K.mean(K.sum(L, 1))




model, eval_model, manipulate_model = CapsNet(input_shape=train_x_temp.shape[1:],
 n_class=len(np.unique(np.argmax(train_y, 1))),
 routings=3)


# compile the model
model.compile(optimizer=optimizers.Adam(lr=0.001),
 loss=[margin_loss, 'mse'],
 loss_weights=[1., 0.392],
 metrics={'capsnet': 'accuracy'})

model.summary()


# Begin: Training with data augmentation ---------------------------------------------------------------------#
def train_generator(x, y, batch_size, shift_fraction=0.1):
 train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
 height_shift_range=shift_fraction) # shift up to 2 pixel for MNIST
 generator = train_datagen.flow(x, y, batch_size=batch_size)
 while 1:
     x_batch, y_batch = generator.next()
     yield ([x_batch, y_batch], [y_batch, x_batch])

# Training with data augmentation. If shift_fraction=0., also no augmentation.
trained_model3 = model.fit_generator(generator=train_generator(train_x_temp, train_y, 1000, 0.1),
 steps_per_epoch=int(train_y.shape[0] / 1000),
 epochs=epochs,
 validation_data=[[val_x_temp, val_y], [val_y, val_x_temp]])
# End: Training with data augmentation -----------------------------------------------------------------------#