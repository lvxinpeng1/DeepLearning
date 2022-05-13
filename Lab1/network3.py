import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 数据准备，构造训练集
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_t = np.arange(-5, 5, 5/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = 30*sigmoid(x_t)

def test_fun():
    
    # 定义模型
    inputs = tf.keras.Input(shape=(3,), name='data')
    outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
    Lm1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='Lm1')
    
    # 定义存储模型的回调函数，请补充完整
    ########## Begin ###########
    checkpoint_path = "./checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=10)
    ########## End ###########
    
    # 编译模型，定义相关属性
    Lm1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),loss='mse', metrics=['mse'])
    
    # 在训练的过程中使用回调函数保存模型，请补充完整
    ########## Begin ###########
    Lm1.fit(x_train, y_train, epochs=200, callbacks=[cp_callback])
    ########## End ###########
    
    loss,acc= Lm1.evaluate(x_train, y_train)
    print("saved model, loss: {:5.2f}".format(loss))
    
    
    # 取出最后一次保存的断点并构建模型加载参数，请补充完整
    ########## Begin ###########
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    Lm2 = tf.keras.Model(inputs=inputs, outputs=outputs, name='Lm2')
    Lm2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),loss='mse', metrics=['mse'])
    Lm2.load_weights(latest)
    
    return Lm2
    ########## End ###########

##################以下将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
# 用模型进行预测，并画出拟合曲线
#loss= Lm2.evaluate(x_train, y_train)
#print("Restored model, loss: {:5.2f}".format(loss))
#
#forecast=Lm2(x_train)
#plt.figure()
#plot1 = plt.plot(x_t, y_train, 'b', label='original values')
#plot2 = plt.plot(x_t, forecast, 'r', label='polyfit values')
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.legend(loc=4)
#plt.savefig('./test_figure/ls_recall/fig.jpg')
