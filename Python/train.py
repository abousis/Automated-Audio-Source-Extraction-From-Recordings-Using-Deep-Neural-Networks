import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from time import time
from data_generator import DataGenerator
from svs_model import svs_model

# Parameters
TIB = 1  # Tracks in batch for random mixing
# TIB = 1 ---> Batch Size = 323 = chunk_per_track
EPOCHS = 15  # Total epochs
SPE = 2000  # Steps per epoch

print('<--[INFO] creating batch generators...')
train_gen = DataGenerator(SPE, TIB, 'train', 'train')
valid_gen = DataGenerator(SPE, TIB, 'train', 'valid')

print('<--[INFO] creating and compiling model...')
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)
accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='binary_crossentropy')
# Importing the model
model = svs_model()
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy_metric, loss])
model.summary()
print('<--[INFO] training network...')
t0 = time()
base_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = "../Models/model.h5"
stats_path = "../Stats/model.csv"
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(base_dir, checkpoint_path), monitor='val_binary_crossentropy', verbose=1,
                                       save_best_only=True, mode='min'),
    tf.keras.callbacks.CSVLogger(filename=os.path.join(base_dir, stats_path),
                                 separator=',', append=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', min_delta=0, patience=3, verbose=0, mode="min",
                                     baseline=None,
                                     restore_best_weights=True)
    ]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=SPE,
    validation_data=valid_gen,
    validation_steps=SPE // 5,  # Validation set is about 1/5 of training set
    callbacks=my_callbacks)
t1 = time()
print("<--[INFO] model was trained in " + str(round((t1 - t0) / 60, 1)) + " minutes")
