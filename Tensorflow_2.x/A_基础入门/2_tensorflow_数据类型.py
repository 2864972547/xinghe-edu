import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# 数据集 = 特征值(float32 - float64) + 目标值(int32)   划分数据集 = 训练特征值+训练目标值  测试特征值+测试目标值  验证特征值+验证目标值
print(tf.constant([[1, 2], [3, 4]]))  # int32  整型
print(tf.constant([[1., 2.0], [3., 4.]])) # float32 浮点型
# 强制转换数据类型  int -> float
# todo 注意点：tensorflow 会根据数据判断数据类型，不同类型不能直接强制转换数据类型 int32 > float32，
#  但是我们同数据类型之间可以强制转换  float 32 > float64 (一维)
# TypeError: Cannot convert 2.0 to EagerTensor of dtype int32
# print(tf.constant(2.,dtype=tf.int32))  # float >>> int
print(tf.constant(2.,dtype=tf.float64))  #float32  >>>  float64
print(tf.constant(2.,dtype=tf.double))  #float32  >>>  float64  [2,4,6,8,10,32,64]

print(tf.constant([True,False]))  # bool 布尔值
print(tf.constant("tensorflow 2.4!")) # str 字符串
# todo  注意点 : 如果实在是需要强制转换的时候，可以使用numpy将图片数据取出出来再强制转换成 EagerTensor
a = np.arange(8)  # 范围生成数组
print(a)  # [0 1 2 3 4 5 6 7]
print(type(a)) # <class 'numpy.ndarray'>
b = tf.cast(a, tf.float64)
print(b)  # tf.Tensor([0. 1. 2. 3. 4. 5. 6. 7.], shape=(8,), dtype=float32)
print(type(b))  # <class 'tensorflow.python.framework.ops.EagerTensor'>

# 高维度数据是否可以直接强制转换数据类型 ，高维度的数据（2D|3D|4D）可以直接强制转换
print(tf.constant([[1, 2], [3, 4]]))
print(tf.constant([[1, 2], [3, 4]],dtype=tf.float64))  # int32