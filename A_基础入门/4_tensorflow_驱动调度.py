import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")   # 忽略警告

print(tf.__version__)  # 查看版本
# print(tf.test.is_gpu_available())   # 查看是否支持显卡
gpus = tf.config.list_physical_devices('GPU')
# gpus = tf.config.list_physical_devices('CPU')
print(gpus)

# todo 1.限制GPU显存使用量
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,enable=True)
    except RuntimeError as e:
        print(e)
# todo 2. 动态分配显存 ->  遇到问题爆显存 -> 设置限制消耗固定大小的显存
print(dir(tf.config))  # 当不清楚这个函数怎么使用的时候，调用内置函数dir()
# 参数1 单张显卡驱动设备 参数2 设置限制消耗固定大小  -> 1024 -> 1G显存
tf.config.set_logical_device_configuration(gpus[0],
                                           [tf.config.LogicalDeviceConfiguration(1024)])  # 限制消耗固定大小的显存
# todo 3.分布式策略
# 3.1定义分布式策略
strategy = tf.distribute.MirroredStrategy()
# 3.2 在上下文环境里面定义分布式模型(神经网络)
with strategy.scope():
    # 定义一层神经网络   Sequential:序贯模型连接所有的神经网络
    tf.keras.Sequential([
        # 参数1:units 神经元个数  参数2：input_shape：图像输入的形状 28 * 28
        # 参数3：softmax 激活函数 将神经网络输出的结果转换成概率
        tf.keras.layers.Dense(units=10, input_shape=(784,), activation='softmax')
    ])

# cd  C:\Windows\System32  -> nvidia-smi.exe
# Fri Mar  8 09:16:53 2024
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 517.00       Driver Version: 517.00       CUDA Version: 11.7     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 Off |                  N/A |
# | N/A   55C    P8     4W /  N/A |    162MiB /  4096MiB |      5%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
#
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# +-----------------------------------------------------------------------------+
