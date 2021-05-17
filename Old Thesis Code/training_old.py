import tensorflow as tf
import numpy as np
import h5py
import os
from generator import generator

# Getting current path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading static preprocessed dataset
data = h5py.File(os.path.join(dir_path, 'mono_dataset_shuffled_ove15.hdf5'), 'r')
train_data = data['train']
train_labels = data['train_label']
test_data = data['test']
test_labels = data['test_label']

# Loading autoencoder model
model = tf.keras.models.load_model('Autoencoder Checkpoints/autoencoder.h5')

# Printing model's layout
model.summary()

# Training callbacks
checkpoint_path = 'Train Checkpoints/model.h5'
my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_mean_squared_error',verbose=1, save_best_only=True, mode='min'),
                tf.keras.callbacks.CSVLogger(filename='Autoencoder Checkpoints/model.csv',
                                             separator=',', append=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0, patience=100, verbose=0, mode="min", baseline=None,
                                                 restore_best_weights=True)
               ]

# Training parameters
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
model.compile(optimizer = optimizer, loss = loss, metrics = loss)

# Defining batch size and data generators
batch_size = 256
training_generator = generator(train_data, train_labels, batch_size)
validation_generator = generator(test_data, test_labels, batch_size)

model.fit(
    training_generator,
    epochs=300,
    steps_per_epoch= train_data.shape[0]/batch_size,
    validation_data=validation_generator,
    validation_steps=test_data.shape[0]/batch_size * 2,
    callbacks=my_callbacks)
