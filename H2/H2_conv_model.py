from keras import backend
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, regularizers, Model, Input

class CNN_for_H2():
    def __init__(self):
        adapter = Model_adapter()
        self.model = Model(inputs=adapter.inputs, outputs=adapter.outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                           loss={'ide': adapter.ide_loss, 'red': adapter.red_loss, 'col': adapter.col_loss},
                           loss_weights=[1, 1, 1])


class Model_adapter:
    zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
    one = tf.convert_to_tensor(1.0, dtype=tf.float32)
    norm = tf.convert_to_tensor(1.0, dtype=tf.float32)
    eps = tf.convert_to_tensor(0.000001, dtype=tf.float32)

    def __init__(self):
        self.inputs = Input(shape=(400, 1), name='input')
        x = layers.Conv1D(100, 32, strides=3, input_shape=(400, 1), activation='relu', kernel_regularizer=regularizers.L2(0.005), bias_regularizer=regularizers.L2(0.005))(self.inputs)
        x = layers.MaxPooling1D(strides=2, pool_size=7)(x)
        x = layers.Dropout(0.02)(x)
        x = layers.Conv1D(96, 16, strides=1, activation='relu', kernel_regularizer=regularizers.L2(0.005), bias_regularizer=regularizers.L2(0.005))(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Dropout(0.02)(x)
        x = layers.Conv1D(96, 16, strides=1, activation='relu', kernel_regularizer=regularizers.L2(0.005), bias_regularizer=regularizers.L2(0.005))(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Dropout(0.02)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(96, kernel_regularizer=regularizers.L2(0.005), bias_regularizer=regularizers.L2(0.005))(x)
        ide = layers.Dense(1, activation='sigmoid', name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        col = layers.Dense(1, name='col')(x)
        self.outputs = [ide, red, col]
