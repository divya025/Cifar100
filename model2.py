# CIFAR 100 Classification Using Keras and Tensor Flow

# Import the libraries required for dong the classification
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import datetime 
import numpy as np

# For graph visualization 
keras.backend.set_image_dim_ordering('tf')
tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                          write_graph=True, write_images=True)


# Constants required for training the model
batch_size = 50
num_classes = 100
epochs = 100
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model2_trained.h5'


# Loda the cifar dataset 
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
print('x_train shape: ', x_train.shape)
print('train samples size: ', x_train.shape[0])
print( 'test samples size: ', x_test.shape[0])

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Convert to ndarray type  float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data Normalization
x_train = x_train / 255.
x_test = x_test / 255.


# Build my model ( Architecture)
model = Sequential()

input_dim = x_train.shape[1:]

# input layer
model.add(Conv2D(32, (3,3),
                 padding='same',
                 input_shape = input_dim))
model.add(Activation('relu'))

# MAXPOOL Layer 1
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

#CONV2D Layer 2
model.add(Conv2D(64, (3,3),
                 padding='same'))
model.add(Activation('relu'))

# MAXPOOL Layer 2
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

# Add a dropout of 10%
model.add(Dropout(0.1))

# CONV2D Layer 3
model.add(Conv2D(128,
		 (3,3), 
		 padding='same'))
model.add(Activation('relu'))

# MAXPOOL Layer 3
model.add(MaxPooling2D(pool_size=(2,2)))

# Add dropout of 25%
model.add(Dropout(0.25))

# flatten 
model.add(Flatten())

# Fully Connected Layer 1
model.add(Dense(512))
model.add(Activation('relu'))

# Adding a dropout of 50%
model.add(Dropout(0.5))

# Output Layer (Fully Connected Layer 2) 
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Adding Adam optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# Compile the Model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Start the timer
start_time = datetime.datetime.now()
print("Start Time is : ",start_time)

# Training the Model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs = epochs,
          validation_split = 0.17,
          shuffle = True,
          callbacks = [tb_callback])

# End the timer
end_time = datetime.datetime.now()
print("End Time is : ",end_time)

total_time = end_time - start_time
print("Total time is :", total_time)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
model.save_weights('model2_weights.h5')
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

from keras.utils import plot_model
plot_model(model, to_file='model.png')
