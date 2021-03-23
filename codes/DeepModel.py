import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, concatenate, Reshape, MaxPool2D, Flatten, Dense, Dropout
tf.keras.backend.set_floatx('float32')


class OneHotConvLayer(layers.Layer):
    def __init__(self, input_shape):
        super().__init__()
        # Inception
        self.conv2d_1_1 = Conv2D(filters=32, kernel_size=(5,4), strides=(1,4), padding='same', activation='relu',
                               name='conv1d_1',input_shape=input_shape)
        self.conv2d_1_2 = Conv2D(filters=32, kernel_size=(4,4), strides=(1,4), padding='same', activation='relu',
                               name='conv1d_2', input_shape=input_shape)
        self.conv2d_1_3 = Conv2D(filters=32, kernel_size=(3,4), strides=(1,4), padding='same', activation='relu',
                               name='conv1d_3', input_shape=input_shape)
        self.reshape_1 = Reshape([101,32,3], name='reshape1', input_shape=[101,3,32])
        # Inception
        self.maxpool_1 = MaxPool2D(pool_size=(2, 2), name='maxpool_1', padding='valid')
        self.conv2d_2 = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', activation='relu',name='conv2d_2')
        self.maxpool_2 = MaxPool2D(pool_size=(2,2), name='maxpool_2',  padding='valid')
        self.conv2d_3 = Conv2D(filters=64, kernel_size=(2, 2), padding='valid', activation='relu',name='conv2d_3')

    def call(self, inputs, training):
        branch_1 = self.conv2d_1_1(inputs, training=training)
        branch_2 = self.conv2d_1_2(inputs, training=training)
        branch_3 = self.conv2d_1_3(inputs, training=training)
        outputs = concatenate([branch_1,branch_2,branch_3],axis=2)
        outputs = self.reshape_1(outputs)

        outputs = self.maxpool_1(outputs, training=training)
        outputs = self.conv2d_2(outputs,training=training)
        outputs = self.maxpool_2(outputs, training=training)
        outputs = self.conv2d_3(outputs, training=training)

        return outputs


class OneHotDeepModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.conv_layer = OneHotConvLayer((101,4,1))
        self.flatten = Flatten(name='flatten')

        self.dropout1 = Dropout(dropout_rate, name="dropout_1")
        self.dense_layer1 = Dense(12, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01),
                                  name="dense_1")

        self.dropout2 = Dropout(dropout_rate, name="dropout_2")
        self.dense_layer2 = Dense(2, activation = "sigmoid", kernel_regularizer=keras.regularizers.l2(0.01),
                                  name="dense_2")

    def call(self, input_x, training):
        y = self.conv_layer(input_x, training=training)
        y = self.flatten(y, training=training)
        y = self.dropout1(y, training=training)
        y = self.dense_layer1(y, training=training)
        y = self.dropout2(y, training=training)
        y = self.dense_layer2(y, training=training)

        return y

    def encode(self,input_x):
        y = self.conv_layer(input_x, training=False)
        y = self.flatten(y, training=False)
        y = self.dropout1(y, training=False)
        y = self.dense_layer1(y, training=False)

        return y


class ChemicaConvLayer(layers.Layer):
    def __init__(self, input_shape):
        super().__init__()
        # Inception
        self.conv2d_1_1 = Conv2D(filters=32, kernel_size=(5,3), strides=(1,3), padding='same', activation='relu',
                               name='conv1d_1',input_shape=input_shape)
        self.conv2d_1_2 = Conv2D(filters=32, kernel_size=(4,3), strides=(1,3), padding='same', activation='relu',
                               name='conv1d_2', input_shape=input_shape)
        self.conv2d_1_3 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,3), padding='same', activation='relu',
                               name='conv1d_3', input_shape=input_shape)
        self.reshape_1 = Reshape([101,32,3], name='reshape1', input_shape=[101,3,32])
        # Inception
        self.maxpool_1 = MaxPool2D(pool_size=(2, 2), name='maxpool_1', padding='valid')
        self.conv2d_2 = Conv2D(filters=32, kernel_size=(2, 2), padding='valid', activation='relu',name='conv2d_2')
        self.maxpool_2 = MaxPool2D(pool_size=(2,2), name='maxpool_2',  padding='valid')
        self.conv2d_3 = Conv2D(filters=64, kernel_size=(2, 2), padding='valid', activation='relu',name='conv2d_3')

    def call(self, inputs, training):
        branch_1 = self.conv2d_1_1(inputs, training=training)
        branch_2 = self.conv2d_1_2(inputs, training=training)
        branch_3 = self.conv2d_1_3(inputs, training=training)
        outputs = concatenate([branch_1,branch_2,branch_3],axis=2)
        outputs = self.reshape_1(outputs)

        outputs = self.maxpool_1(outputs, training=training)
        outputs = self.conv2d_2(outputs,training=training)
        outputs = self.maxpool_2(outputs, training=training)
        outputs = self.conv2d_3(outputs, training=training)

        return outputs


class ChemicalDeepModel(tf.keras.Model):
    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.conv_layer = ChemicaConvLayer((101,3,1))
        self.flatten = Flatten(name='flatten')

        self.dropout1 = Dropout(dropout_rate, name="dropout_1")
        self.dense_layer1 = Dense(12, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01),
                                  name="dense_1")

        self.dropout2 = Dropout(dropout_rate, name="dropout_2")
        self.dense_layer2 = Dense(2, activation = "sigmoid", kernel_regularizer=keras.regularizers.l2(0.01),
                                  name="dense_2")

    def call(self, input_x, training):
        y = self.conv_layer(input_x, training=training)
        y = self.flatten(y, training=training)
        y = self.dropout1(y, training=training)
        y = self.dense_layer1(y, training=training)
        y = self.dropout2(y, training=training)
        y = self.dense_layer2(y, training=training)

        return y

    def encode(self,input_x):
        y = self.conv_layer(input_x, training=False)
        y = self.flatten(y, training=False)
        y = self.dropout1(y, training=False)
        y = self.dense_layer1(y, training=False)

        return y