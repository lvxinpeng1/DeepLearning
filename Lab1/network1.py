import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 数据准备，构造训练集

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_t = np.arange(-5, 5, 5/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis = 1)
y_train = 30*sigmoid(x_t)

def test_fun():
    # 模型定义，请补全构建模型的代码
    ########## Begin ###########
    # 步骤1 定义输入层
    inputs =
    # 步骤2 定义输出层，提供线性回归的参数
    outputs =
    # 步骤3 创建模型包装输入层和输出层
    Lm1 = tf.keras.Model()
    ########## End ###########
    
    # 模型训练，请补全 api 中的参数，从而进行训练
    ########## Begin ###########
    # 步骤1 模型编译
    Lm1.compile()
    # 步骤2 模型训练
    Lm1.fit()
    
    return Lm1
    ########## End ###########


##################以下代码将出现在测试文件中，这里不需要运行，请勿取消注释！！！！！！！！！！！！！！！
# 模型预测
#loss,acc= Lm1.evaluate(x_train, y_train)
#
## 画出拟合曲线
#forecast=Lm1(x_train)
#plt.figure()
#plot1 = plt.plot(x_t, y_train, 'b', label='original values')
#plot2 = plt.plot(x_t, forecast, 'r', label='polyfit values')
#plt.xlabel('x axis')
#plt.ylabel('y axis')
#plt.legend(loc=4)
#plt.savefig('./test_figure/train_api/fig.jpg')
