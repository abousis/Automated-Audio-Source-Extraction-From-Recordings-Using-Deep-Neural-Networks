import tensorflow as tf
from tensorflow.keras import Model, layers


def svs_model():
    """
    :return: Singing Voice Separation Model using Keras Functional API
    """
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
    outputs = layers.Dense(18441, use_bias=True, bias_initializer=tf.constant_initializer(0.1), activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model
