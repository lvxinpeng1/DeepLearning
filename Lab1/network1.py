import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
import os
from pathlib import Path

# 数据准备，构造训练集
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_t = np.arange(-5, 5, 5/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = 30*sigmoid(x_t)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)

def test_fun():
    
    # 定义模型
    inputs = tf.keras.Input(shape=(3,), name='data')
    outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')
    
    # 定义损失函数、优化器、训练损失评估、测试损失评估等训练过程中的必要属性
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(0.05)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    
    # 定义训练过程的静态计算图
    @tf.function
    def train_step(data, labels):
        with tf.GradientTape() as tape:
            predictions = model(data)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        train_loss(loss)
    
    # 定义测试过程的静态计算图
    @tf.function
    def test_step(data, labels):
        predictions = model(data)
        t_loss = loss_object(labels, predictions)
    
        test_loss(t_loss)
    
    # 定义训练迭代次数
    EPOCHS = 220
    
    # 自定义循环进行训练
    for epoch in range(EPOCHS):
        start = time.time()
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        test_loss.reset_states()
    
        for data, labels in train_dataset:
            train_step(data, labels)
    
        for test_data, test_labels in train_dataset:
            test_step(test_data, test_labels)
    
        end = time.time()
        template = 'Epoch {}, Loss: {:.3f},  Test Loss: {:.3f}，Time used: {:.2f}'
        print(template.format(epoch + 1,
                             train_loss.result(),
                             test_loss.result(), end - start))
            
    
    # 将模型文件保存为 "./save_model/model.h5"
    ########## Begin ###########
    model.save("./save_model/model.h5")
    ########## End ###########
    
    
    # 从 "./save_model/model.h5" 加载模型文件并赋值给 model_load
    ########## Begin ###########
    model_load = tf.keras.models.load_model('./save_model/model.h5')
    model_load.compile(loss='mse')
    return model_load
    ########## End ###########


##################以下将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
#loss= model_load.evaluate(x_train, y_train)
#predictions = model_load(x_train)
#plt.figure()
#plot1 = plt.plot(x_t, y_train, 'b', label='original values')
#plot2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.legend(loc=4)
#plt.savefig('./test_figure/ls_general/fig.jpg')
