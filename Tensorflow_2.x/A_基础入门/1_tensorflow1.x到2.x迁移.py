import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")   # 忽略警告

# pip install tensorflow==2.4.0  (学校机房安装)
# pip install tensorflow-gpu==2.4.0 -i http://mirrors.aliyun.com/pypi/simple/ (笔记本安装) -> True

print(tf.__version__)  # 查看版本
print(tf.test.is_gpu_available())   # 查看是否支持显卡

# tensorflow 1.x
# tf.compat.v1.disable_eager_execution()  # 禁用急切模式(将tensorflow切换回1版本)
print(dir(tf.compat.v1))  # 当不清楚这个函数怎么使用的时候，调用内置函数dir()
a = tf.constant([[1, 2], [3, 4]])  # 常数
# tf.compat.v1.InteractiveSession()  # 开启上下文会话
# print(a.eval())

# tensorflow 2.x
print(a)