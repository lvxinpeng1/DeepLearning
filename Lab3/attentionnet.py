import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate, Input, Activation, add, GlobalAveragePooling2D, \
    Reshape, Multiply
from tensorflow.keras.utils import plot_model


def dataModelPreparation():
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

    n_classes = 10

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    yy = y_test
    y_ind = yy.argsort()

    label_mask = np.unique(y_test)
    label_count = {}
    for v in label_mask:
        label_count[v] = np.sum(y_test == v)
    print("label_mask值为：")
    print(label_mask)
    print("统计结果：")
    print(label_count)

    example_images = []
    total_label = 0
    for i in range(0, 10):
        example_images += [x_test[y_ind[total_label:total_label + 10]]]
        total_label += label_count[i]
    # example_images = [x_test[y_ind[i*1100:i*1100+100]] for i in range(0, 10)]
    example_images = np.vstack(example_images)

    # Work 1： 建立one-hot标签及训练和测试数据，注意，batchsize为100###
    y_train = tf.one_hot(y_train, n_classes)
    y_test = tf.one_hot(y_test, n_classes)

    train_datasets = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(100)
    # train_datasets = x_train.batch(100)

    test_datasets = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

    # test_datasets = x_test.batch(100)
    ################################################################

    # Work 2： 建立通道注意力模块#######################################
    def SeNetBlock(input, reduction=4):
        # 得到input的通道数量c
        channels = input.shape[-1]
        # 先对feature的每个通道进行全局平均池化Global Average Pooling 得到通道描述子（Squeeze）
        avg_x = tf.keras.layers.GlobalAveragePooling2D()(input)
        # 形状匹配做点啥？考虑global average pooling后的特征形状是否适合充当conv2d的输入
        avg_x = tf.keras.layers.Reshape((-1, 1, channels))(avg_x)
        # 接着做reduction，用int(channels)//reduction个卷积核对 avg_x做1x1的点卷积, 其他参数：relu,valid
        x = tf.keras.layers.Conv2D(int(channels) // reduction, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='valid')(avg_x)
        # 接着用int(channels)个卷积核个数对 x做1x1的点卷积，扩展x回到原来的通道个数,其他参数：不激活,valid
        x = tf.keras.layers.Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        # 对x 做 sigmoid 激活得到通道权重，用'hard_sigmoid'
        cbam_feature = tf.keras.layers.Activation('hard_sigmoid', name='activation')(x)
        return input, cbam_feature  # 返回以cbam_feature 为scale，对feature做拉伸加权的结果（Excitation）

    ####################################################################

    inputs = tf.keras.Input(shape=(14, 14, 1), name='data')

    x = tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', padding='valid')(inputs)
    x, cbam_feature = SeNetBlock(x, reduction=4)  # SeNetBlock 对有16个卷积核的第二个卷积层进行加权操作

    # Work 3： 将通道注意力模块连入网络#################

    ##################################################
    # x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(x)
    # x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x)

    x = tf.keras.layers.Multiply()([x, cbam_feature])
    x = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    fc1 = tf.keras.layers.Dense(60, activation='relu', name='dense')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='dense_1')(fc1)

    # Work 4： 利用tf.keras.Model建立两个模型，一个为完整的网络，用于训练；另一个用于输出通道注意力特征
    senet_lenet = tf.keras.Model(inputs=inputs, outputs=outputs, name='senet_lenet')
    cbam_feature_model = tf.keras.Model(inputs=inputs, outputs=cbam_feature, name="cbam_feature_model")
    ############################################################################################

    cbam_feature_output = cbam_feature_model(example_images).numpy()
    cbam_feature_matrix = cbam_feature_output.reshape((100, 16))

    plt.figure(num='cbam', figsize=(3, 8))
    plt.imshow(cbam_feature_matrix, cmap=plt.get_cmap('hot'))
    plt.colorbar()
    plt.savefig("./test_figure/step1/cbam.png")

    plot_model(senet_lenet, to_file='./test_figure/step1/senet_lenet.png', show_shapes=True, expand_nested=True)

    return train_datasets, test_datasets, senet_lenet, example_images


def test_fun():
    train_datasets, test_datasets, senet_lenet, example_images = dataModelPreparation()
    # Work 5： 完成模型的编译和训练###########################################
    senet_lenet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['acc']
    )

    history = senet_lenet.fit(
        train_datasets, validation_data=test_datasets, batch_size=100,
        epochs=1  # epoch数量不要改变！！！
    )
    #########################################################################
    return senet_lenet, history
