import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
import os

# 数据准备，构造训练集

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_t = np.arange(-5, 5, 5/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = 30*sigmoid(x_t)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)

def test_fun():
    ########## Begin ###########
    # 步骤1 定义输入层
    inputs = tf.keras.Input(shape=(3,), name='data')
    # 步骤2 定义输出层，提供线性回归的参数
    outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
    # 步骤3 创建模型包装输入层和输出层
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')
    ########## End ###########
    

    # 定义损失函数、优化器及损失计算方法，请补充
    ########## Begin ###########
    # 创建均方误差损失函数
    loss_object =
    # 创建优化器
    optimizer =
    # 创建训练损失计算方法
    train_loss =
    # 创建测试损失计算方法
    test_loss =
    ########## End ###########
    
    
    # 定义 python 函数封装训练迭代过程，请补充
    ########## Begin ###########
    @tf.function
    def train_step(data, labels):

    ########## End ###########
    
    
    # 定义 python 函数为静态图封装测试迭代过程，请补充
    ########## Begin ###########
    @tf.function
    def test_step(data, labels):

    ########## End ###########
    
    
    EPOCHS = 50
    
    # 自定义循环对模型进行训练
    ########## Begin ###########
    for epoch in range(EPOCHS):
        start = time.time()
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()
        test_loss.reset_states()
    
        # 模型在训练集上进行一次训练，请补充
        for data, labels in train_dataset:


        for test_data, test_labels in train_dataset:

            
    
        end = time.time()
        # 输出训练情况
        template = 'Epoch {}, Loss: {:.3f},  Test Loss: {:.3f}，Time used: {:.2f}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              test_loss.result(), end - start))
    return model
    ########## End ###########


##################以下将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
# 模型预测，并画出拟合曲线
#model.compile(loss='mse')
#loss = model.evaluate(x_train, y_train)
#predictions = model(x_train)
#plt.figure()
#plot1 = plt.plot(x_t, y_train, 'b', label='original values')
#plot2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.legend(loc=4)
#plt.savefig('./test_figure/train_loop/fig.jpg')
