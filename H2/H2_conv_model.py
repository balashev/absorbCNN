from keras import backend
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, regularizers, Model, Input

class CNN_for_H2():
    def __init__(self):
        adapter = Model_adapter()
        self.model = Model(inputs=adapter.inputs, outputs=adapter.outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                           loss={'ide': adapter.ide_loss, 'red': adapter.red_loss},
                           loss_weights=[1, 1],
                           metrics={'ide': [adapter.BinaryFalsePositives(), adapter.BinaryFalseNegatives()]}
                           )

class Model_adapter:
    zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
    one = tf.convert_to_tensor(1.0, dtype=tf.float32)
    norm = tf.convert_to_tensor(1.0, dtype=tf.float32)
    eps = tf.convert_to_tensor(0.000001, dtype=tf.float32)

    def __init__(self):
        self.inputs = Input(shape=(64, 6, 1), name='input')
        x = layers.Conv2D(32, (12, 1), input_shape=(64, 6, 1), activation='relu')(self.inputs)
        x = layers.Dropout(0.02)(x)
        x = layers.MaxPooling2D(pool_size=3)(x)
        x = layers.Conv2D(64, (6, 1), activation='relu')(x)
        x = layers.Dropout(0.02)(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(64, (6, 1), activation='relu')(x)
        x = layers.Dropout(0.02)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64)(x)
        x = layers.Dropout(0.02)(x)
        ide = layers.Dense(1, activation='sigmoid', name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        self.outputs = [ide, red]

    def ide_loss(self, y_true, y_pred):
        y_new = tf.reshape(y_true[:,0], shape=tf.shape(y_pred))
        y_pred_pe = tf.add(y_pred, self.eps)
        y_pred_me = tf.subtract(y_pred, self.eps)
        a = tf.subtract(self.zero, tf.math.multiply(y_new, tf.math.log(y_pred_pe)))
        y_true_subtract = tf.subtract(self.one, y_new)
        y_pred_subtract = tf.subtract(self.one, y_pred_me)
        b = tf.math.multiply(y_true_subtract, tf.math.log(y_pred_subtract))
        Lc = tf.subtract(a, b)
        # return backend.sum(Lc, axis=-1)
        return backend.mean(Lc, axis=-1)
        

    def red_loss(self, y_true, y_pred):
        y_new = tf.reshape(y_true[:,1], shape=tf.shape(y_pred))
        return backend.mean(tf.math.squared_difference(y_pred, y_new), axis=-1)
        # return backend.sum(tf.math.squared_difference(y_pred, y_new), axis=-1)

    def col_loss(self, y_true, y_pred):
        y_ide = tf.reshape(y_true[:,0], shape=tf.shape(y_pred))
        y_new = tf.reshape(y_true[:,2], shape=tf.shape(y_pred))
        denominator = tf.math.add(y_ide, self.eps)
        fraction = tf.math.divide(y_ide, denominator)
        squared_diff = tf.math.squared_difference(y_new, y_pred)
        Lh = tf.math.multiply(fraction, squared_diff)
        # return backend.sum(Lh, axis=-1)
        return backend.mean(Lh, axis=-1)

    class BinaryTruePositives(metrics.Metric):               #example

      def __init__(self, name='binary_true_positives', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(tf.reshape(y_true[:,0], shape=tf.shape(y_pred)), tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(tf.round(tf.reshape(y_true[:,0], shape=tf.shape(y_pred))), tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives

      def reset_state(self):
        self.true_positives.assign(0)

    class BinaryTrueNegatives(metrics.Metric):

      def __init__(self, name='binary_true_negatives', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.round(tf.reshape(y_true[:,0], shape=tf.shape(y_pred))), tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives

      def reset_state(self):
        self.true_positives.assign(0)

    class BinaryFalsePositives(metrics.Metric):

      def __init__(self, name='binary_false_positives', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.round(tf.reshape(y_true[:,0], shape=tf.shape(y_pred))), tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives

      def reset_state(self):
        self.true_positives.assign(0)

    class BinaryFalseNegatives(metrics.Metric):

      def __init__(self, name='binary_false_negatives', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

      def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.round(tf.reshape(y_true[:,0], shape=tf.shape(y_pred))), tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
          sample_weight = tf.cast(sample_weight, self.dtype)
          values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

      def result(self):
        return self.true_positives

      def reset_state(self):
        self.true_positives.assign(0)
