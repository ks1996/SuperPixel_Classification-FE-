from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import losses
from keras import optimizers
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#K.set_image_dim_ordering('tf')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from time import time
import cv2
import numpy as np
from random import shuffle
from keras.callbacks import TensorBoard
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
'''
import os
import tensorflow as tf
from PIL import ImageFile

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
nb_train_samples = 17000
nb_validation_samples = 200
img_width, img_height =56,56
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_data = "/nfs001/kavya/inputData/NEW_SP"
#test_data = "/nfs001/joshua/2016data/test"
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=32,
	class_mode='categorical',
        )
'''
validation_generator = test_datagen.flow_from_directory(
        test_data,
        target_size=(img_width, img_height),
        batch_size=32,
	class_mode='categorical',
       )
'''

epochs = 300
batch_size = 32




model = Sequential()

#1st Stage

model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',input_shape=(img_width,img_height,3))) #56x56x3
'''  
model.add(Flatten())#N, 46656
model.add(Dense(2916, init='uniform',activation='relu'))#N,2916
model.add(Dropout(0.25))#N,2916
model.add(Reshape((54,54,1)))#N, 54x54x16
'''

model.add(Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation='relu',padding='same'))#N, 54x54x16
'''
model.add(Flatten())#N, 46656
model.add(Dense(2916, init='uniform',activation='relu'))#N,2916
model.add(Dropout(0.25))#N,2916
model.add(Reshape((54,54,1)))#N, 54x54x1
'''
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))


model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))


#2nd Stage

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))
'''
model.add(Flatten())#N, 23328
model.add(Dense(729, init='uniform',activation='relu'))#N,2916
model.add(Dropout(0.25))#N,729
model.add(Reshape((27,27,1)))#N, 27x27x32
'''

model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1),padding='same',
                 activation='relu'))#N, 27x27x32
'''
model.add(Flatten())#N, 23328
model.add(Dense(729, init='uniform',activation='relu'))#N,729
model.add(Dropout(0.25))#N,729
model.add(Reshape((27,27,1)))#N, 27x27x32

'''
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))


model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))



#3rd Stage

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))#14x14x64
'''
model.add(Flatten())#N, 12544
model.add(Dense(196, init='uniform',activation='relu'))#N,196
model.add(Dropout(0.25))#N,729
model.add(Reshape((14,14,1)))#N, 14x14x64
'''
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1),padding='same',
                 activation='relu'))
'''
model.add(Flatten())#N, 12544
model.add(Dense(196, init='uniform',activation='relu'))#N,196
model.add(Dropout(0.25))#N,729
model.add(Reshape((14,14,1)))#N, 14x14x64
'''
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))


model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'))


#4th Stage


model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))#7x7x128
'''
model.add(Flatten())#N, 6272
model.add(Dense(49, init='uniform',activation='relu'))#N,49
model.add(Dropout(0.25))#N,49
model.add(Reshape((7,7,1)))#N, 7x7x128
'''
model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1),padding='same',
                 activation='relu'))
'''
model.add(Flatten())#N, 6272
model.add(Dense(49, init='uniform',activation='relu'))#N,49
model.add(Dropout(0.25))#N,49
model.add(Reshape((7,7,1)))#N, 7x7x128

'''
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),padding='same',
                 activation='relu'))



model.add(AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same', data_format=None))# (None, 7, 7, 128) 


model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4 ,activation='softmax'))




sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

#tensorboard1 = TensorBoard(log_dir="logs/{}".format(time()))

tensorboard = TensorBoard(log_dir='/nfs001/kavya/inputData/logs/', histogram_freq=0, write_graph=True, write_images=True)


# define the checkpoint
filepath = "/nfs001/kavya/inputData/file.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[tensorboard])
   # validation_data=validation_generator,
    #validation_steps=nb_validation_samples // batch_size)


model.save_weights('first_try.h5') 

model.summary()






