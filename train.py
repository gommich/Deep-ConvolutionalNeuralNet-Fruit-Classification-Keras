#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
import time
import tensorflow
import numpy as np
import random
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,Flatten,Dropout,BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tensorflow.set_random_seed(SEED)


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """

    ht = 150
    wd = 150
    ins = (ht,wd,3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = ins, activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = "relu")) 
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = "softmax", kernel_regularizer=regularizers.l2(0.01)))


    return model


def train_and_save_model(model):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """

    ht = 150
    wd = 150
    ins = (ht,wd,3)


    image_train_generator = ImageDataGenerator(
	rescale=1./255,
	validation_split=0.10,
	shear_range = 0.2,
	zoom_range=0.2,
	horizontal_flip=True
		
    )

    train_generator = image_train_generator.flow_from_directory(
	'data/train',
	target_size=(ht,wd),
	color_mode='rgb',
	batch_size=100,
	shuffle=True,
	subset='training'
     )

    validation_generator = image_train_generator.flow_from_directory(
	'data/train',
	target_size=(ht,wd),
	color_mode='rgb',
	batch_size=100,
	shuffle=True,
	subset='validation'
	)

    opt = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])

    model.summary()

    hist=keras.callbacks.History()
    checkpoint=keras.callbacks.ModelCheckpoint(filepath='model/model.h5', monitor='val_acc',    	save_best_only=True)

    start = time.time()

    model.fit_generator(
	train_generator,
	steps_per_epoch=train_generator.samples/100,
	validation_data=validation_generator,
	validation_steps=validation_generator.samples/100,
	epochs=1,
	callbacks=[checkpoint,hist]
    )

    end = time.time()
    print("Time Taken:", end - start)




if __name__ == '__main__':
    model = construct_model()
    model = train_and_save_model(model)
    
