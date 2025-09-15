from keras import activations, Input, layers, metrics, Model, models, optimizers, regularizers
import tensorflow as tf
#from tensorflow.keras import models, optimizers, metrics, regularizers, Model, Input

class CNN_for_DLA_keras():
    def __init__(self, dt='float32'):
        self.dt = getattr(tf, dt)
        tf.keras.backend.set_floatx(dt)
        adapter = Model_adapter(dt=self.dt)
        self.model = Model(inputs=adapter.inputs, outputs=adapter.outputs)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                           loss={'ide': adapter.ide_loss, 'red': adapter.red_loss, 'col': adapter.col_loss},
                           loss_weights=[1.0, 1.0, 1.0],
                           #metrics = ['accuracy']
                           metrics={'ide': [adapter.BinaryTruePositives(), adapter.BinaryFalsePositives(), adapter.BinaryFalseNegatives()]}
                           )

    def predict(self, specs):
        return self.model.predict(specs)

class Model_adapter:
    def __init__(self, dt):
        self.zero = tf.convert_to_tensor(0.0, dtype=dt)
        self.one = tf.convert_to_tensor(1.0, dtype=dt)
        self.conf = tf.convert_to_tensor(0.5, dtype=dt)
        self.eps = tf.convert_to_tensor(0.000001, dtype=dt)
        self.inputs = Input(shape=(400, 1), name='input')
        self.define_model()

    def define_model_simplest(self, dropout=0.1, regul=0.005, activation='elu'):
        x = layers.Conv1D(50, 16, strides=5, input_shape=(400, 1), activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul), activity_regularizer=regularizers.L1L2(regul))(self.inputs)
        print("conv1d:", x.shape)
        x = layers.Dropout(dropout)(x)
        print("dropout:", x.shape)
        x = layers.MaxPooling1D(strides=3, pool_size=5)(x)
        print("maxpooling:", x.shape)
        x = layers.Flatten()(x)
        print("flatten:", x.shape)
        x = layers.Dense(96, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul), activity_regularizer=regularizers.L1L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        ide = layers.Dense(1, activation=activation, name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        #col = layers.Dense(1, activation='relu', name='col')(x)
        col = layers.Dense(1, name='col')(x)
        self.outputs = [ide, red, col]

    def define_model_reduced(self, dropout=0.1, regul=0.005, activation='elu'):
        x = layers.Conv1D(100, 32, strides=3, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(self.inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=2, pool_size=7)(x)
        x = layers.Conv1D(96, 16, strides=1, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=1, pool_size=7)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(96, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        ide = layers.Dense(1, activation='sigmoid', name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        #col = layers.Dense(1, activation='relu', name='col')(x)
        col = layers.Dense(1, name='col')(x)
        self.outputs = [ide, red, col]

    def define_model(self, dropout=0.1, regul=0.005, activation='elu'):
        x = layers.Conv1D(100, 32, strides=3, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(self.inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=2, pool_size=7)(x)
        x = layers.Conv1D(96, 16, strides=1, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Conv1D(96, 16, strides=1, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(96, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        ide = layers.Dense(1, activation='sigmoid', name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        #col = layers.Dense(1, activation='relu', name='col')(x)
        col = layers.Dense(1, name='col')(x)
        self.outputs = [ide, red, col]

    def define_model_backup(self, dropout=0.1, regul=0.005, activation='elu'):
        self.inputs = Input(shape=(400, 1), name='input')
        x = layers.Conv1D(100, 32, strides=3, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(self.inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=2, pool_size=7)(x)
        x = layers.Conv1D(96, 16, strides=1, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Conv1D(96, 16, strides=1, activation=activation, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.MaxPooling1D(strides=1, pool_size=6)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(96, kernel_regularizer=regularizers.L2(regul), bias_regularizer=regularizers.L2(regul))(x)
        x = layers.Dropout(dropout)(x)
        ide = layers.Dense(1, activation='sigmoid', name='ide')(x)
        red = layers.Dense(1, name='red')(x)
        #col = layers.Dense(1, activation='relu', name='col')(x)
        col = layers.Dense(1, name='col')(x)
        self.outputs = [ide, red, col]

    def ide_loss(self, y_true, y_pred):
        y_new = tf.reshape(y_true[:,0], shape=tf.shape(y_pred))
        a = tf.subtract(self.zero, tf.math.multiply(y_new, tf.math.log(tf.add(y_pred, self.eps))))
        b = tf.math.multiply(tf.subtract(self.one, y_new), tf.math.log(tf.subtract(self.one, tf.subtract(y_pred, self.eps))))
        return tf.math.reduce_sum(tf.subtract(a, b), axis=-1)

    def red_loss(self, y_true, y_pred):
        y_ide = tf.reshape(y_true[:,0], shape=tf.shape(y_pred))
        y_new = tf.reshape(y_true[:,1], shape=tf.shape(y_pred))
        #return tf.math.reduce_mean(tf.math.squared_difference(y_pred, y_new), axis=-1)
        return tf.math.reduce_sum(tf.math.multiply(tf.math.divide(y_ide, tf.math.add(y_ide, self.eps)), tf.math.squared_difference(y_pred, y_new)), axis=-1)

    def col_loss(self, y_true, y_pred):
        y_ide = tf.reshape(y_true[:,0], shape=tf.shape(y_pred))
        y_new = tf.reshape(y_true[:,2], shape=tf.shape(y_pred))
        return tf.math.reduce_sum(tf.math.multiply(tf.math.divide(y_ide, tf.math.add(y_ide, self.eps)), tf.math.squared_difference(y_new, y_pred)), axis=-1)

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