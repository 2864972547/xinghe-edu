import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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
def set_GPU():
    """GPU相关配置"""
    # todo 2.1 打印变量在哪个设备上运行
    print(tf.debugging.set_log_device_placement(True))
    # todo 2.2 获取当前设别GPU的个数
    cpus = tf.config.experimental.list_physical_devices('CPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(cpus)
    print(f"物理机CPU的个数为：{len(cpus)} ，物理GPU的个数为: {len(gpus)}" )
    # todo 2.3 设置显存自增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,enable=True)
    # todo 2.4 设置GPU设备可见可用,当不清楚当前服务器有多少设备的情况下，设置-1自动匹配
    tf.config.experimental.set_visible_devices(gpus[-1],'GPU')
    # todo 2.5 切分逻辑GPU
    # 假设当前显存为4G  -> 1G=1024显存进行加载数据 2G=2048显存进行模型训练
    print(dir(tf.config.experimental))
    # 切割GPU个数和每个逻辑GPU的大小
    tf.config.experimental.set_virtual_device_configuration(
        gpus[-1],
        [tf.config.experimental.VirtualDeviceConfiguration(1024),
        tf.config.experimental.VirtualDeviceConfiguration(2048),
    ])
    # todo 2.6 获取逻辑GPU个数
    logical_devices = tf.config.experimental.list_logical_devices('GPU')
    print("逻辑GPU个数为：{}".format(len(logical_devices)))
    # todo 2.7 手动指定多GPU环境
    c = []
    for gpu in logical_devices:
        print(gpu.name)
        # with tf.device(gpu.device):
        with tf.device("/device:GPU:0"):
            a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])  # (2,3)
            b = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]]) # (3,2)
            c.append(tf.matmul(a, b))
        with tf.device("/device:CPU:0"):
            matmul_sum = tf.add_n(c)
            print(matmul_sum)
    print(c)

if __name__ == '__main__':
    set_GPU()
# Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
# 缺少动态库  ->cd  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin
#  cusolver64_11.dll 修改成 cusolver64_10.dll 重新粘贴进 bin目录即可