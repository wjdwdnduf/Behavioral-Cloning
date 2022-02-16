import numpy as np
import cv2
import sklearn
import csv
import math
import tensorflow as tf
import os

samples = []
with open ('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for index in range(0, 3):
                    name = './mydata/IMG/'+batch_sample[index].split('/')[-1]
                    image = cv2.imread(name)
                    center_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if index == 0:
                        center_angle = float(batch_sample[3])
                    # create adjusted steering measurements for the side camera images
                    elif index == 1:
                        center_angle = float(batch_sample[3]) + 0.2
                    else:
                        center_angle = float(batch_sample[3]) - 0.2
                        
                    # Flipping Images And Steering Measurements    
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
   
                    # Center image and angle
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)
            
# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Lambda
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape =(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
          
# Add Convolutions and dropouts here
model.add(Convolution2D(24,5,5,subsample=(2,2))) # stride: (2,2)
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2))) # stride: (2,2)
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2))) # stride: (2,2)
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100)) 
model.add(Activation('relu'))
model.add(Dense(50)) 
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1)) 
model.compile(loss='mse',optimizer='adam')
# Fit data here
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs = 7, verbose = 1)
# Save model
model.save('model.h5')