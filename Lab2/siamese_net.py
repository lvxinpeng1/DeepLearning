import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import time
import os
num_classes = 10
#1. Data preparation
def dataPreparation():
    path = './dataset/mnist.npz'
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    h = x_train.shape[1] // 2
    w = x_train.shape[2] // 2

    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [h, w]).numpy()  # if we want to resize
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = tf.image.resize(x_test, [h, w]).numpy()  # if we want to resize

    print(x_train.shape)

    # 1. prepare datasets
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # TODO: Add your codes here
   
    # (1) 将图片分割为上下两个部分：x_train1, x_train2, x_test1, x_test2
    x_train1, x_train2 = tf.split(x_train, num_or_size_splits=2, axis = 1)
    x_test1, x_test2 = tf.split(x_test, num_or_size_splits=2, axis = 1)

    # (2) 将标签转换为one_hot格式
    y_train = tf.one_hot(y_train, num_classes)
    y_test = tf.one_hot(y_test, num_classes)

    len_train1 = len(x_train1)
    len_test1 = len(x_test1)
    len_train2 = len(x_train2)
    len_test2 = len(x_test2)

    x_train1 = tf.reshape(x_train1, [len_train1, -1])
    x_test1 = tf.reshape(x_test1, [len_test1, -1])
    x_train2 = tf.reshape(x_train2, [len_train2, -1])
    x_test2 = tf.reshape(x_test2, [len_test2, -1])
    

    # (3) 利用tf.data.Dataset中的工具生成数据集
    train_datasets = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((x_train1, x_train2)),tf.data.Dataset.from_tensor_slices(y_train))).batch(64)
    test_datasets = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((x_test1, x_test2)), tf.data.Dataset.from_tensor_slices(y_test))).batch(64)

    return train_datasets, test_datasets

class BiasPlusLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        super(BiasPlusLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.bias = self.add_weight("bias",shape=[self.num_outputs])
    def build(self, input_shape):
        super(BiasPlusLayer, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, input):
        return input[0]+input[1] + self.bias
#2. net_build
def BuildModel():
    # TODO: Add your codes here
    shared_base = tf.keras.Sequential([
        Input(shape=(98,),name = 'D1_input'),
        Dense(64, activation='relu',name='D1')
        ], name = 'seq1')

    x1 = Input(shape=(98,), name='Input1')
    x2 = Input(shape=(98,), name='Input2')
    b1 = shared_base(x1)
    b2 = shared_base(x2)

    b = BiasPlusLayer(64,name='BiasPlusLayer')([b1,b2])

    outputs = Dense(num_classes, activation='softmax')(b)

    siamese_net = Model(inputs=[x1,x2], outputs = outputs)

    tf.keras.utils.plot_model(siamese_net, to_file='./test_figure/step1/siamese_net.png', show_shapes=True, expand_nested=True)

    return siamese_net


#3. train and test
def test_fun():
    siamese_net = BuildModel()
    train_datasets, test_datasets = dataPreparation()
    # TODO: Add your codes here
    siamese_net.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['acc'])

    history = siamese_net.fit(train_datasets, epochs=5, validation_data=test_datasets)
    return siamese_net, history

















