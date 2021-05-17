import tensorflow as tf
import h5py
import os
from tensorflow.keras import layers, Model
from generator import generator

# Getting current path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Loading static preprocessed dataset
data = h5py.File(os.path.join(dir_path, 'mono_dataset_shuffled_ove15.hdf5'), 'r')
train_data_AE = data['train_auto_encoder']
train_labels = data['train_label']
test_data_AE = data['test_auto_encoder']
test_labels = data['test_label']

# Defining model using Keras Functional API
inputs = layers.Input(shape=18441)
x = layers.Reshape(target_shape=[9, 2049, 1])(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3, 12), kernel_initializer='glorot_uniform', use_bias=True,
                    bias_initializer=tf.constant_initializer(0.1), padding='same', activation='relu')(x)
x = layers.Conv2D(filters=16, kernel_size=(3, 12), use_bias=True, bias_initializer=tf.constant_initializer(0.1),
                    padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(1, 12), padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=(3, 12), use_bias=True, bias_initializer=tf.constant_initializer(0.1),
                    padding='same', activation='relu')(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 12), use_bias=True, bias_initializer=tf.constant_initializer(0.1),
                    padding='same', activation='relu')(x)
x = layers.MaxPool2D(pool_size=(1, 12), padding='same')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(2048, use_bias=True, bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(512, use_bias=True, bias_initializer=tf.constant_initializer(0.1), activation='relu')(x)
outputs = layers.Dense(18441, use_bias=True, bias_initializer=tf.constant_initializer(0.1), activation=None)(x)
model = Model(inputs, outputs)

# Printing model's layout
model.summary()

# Training callbacks
checkpoint_path = "Autoencoder Checkpoints/autoencoder.h5"
my_callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_mean_squared_error',verbose=1, save_best_only=True, mode='min'),
                tf.keras.callbacks.CSVLogger(filename='Autoencoder Checkpoints/autoencoder.csv',
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
training_generator = generator(train_data_AE, train_labels, batch_size)
validation_generator = generator(test_data_AE, test_labels, batch_size)

model.fit(
    training_generator,
    epochs=250,
    steps_per_epoch= train_data_AE.shape[0]/batch_size,
    validation_data=validation_generator,
    validation_steps=test_data_AE.shape[0]/batch_size * 2,
    callbacks=my_callbacks)
