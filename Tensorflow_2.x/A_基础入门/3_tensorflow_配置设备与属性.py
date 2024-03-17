import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # 禁用急切模式(将tensorflow切换回1版本)
# Tensorflow 1.x 版本属性
with tf.Graph().as_default():   # 通过上下文环境开启默认图
    with tf.device('/cpu:0'):  # 通过上下文环境设置驱动程序运行具体tensor
        a = tf.constant(1.0, dtype=tf.float32)
        b = tf.constant([1.0,2.0], dtype=tf.float32)
    with tf.device('/gpu:0'):  # 设置驱动程序运行具体tensor
        c = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        d = tf.constant([[[1, 2], [3, 4]]], dtype=tf.float32)
        e = tf.constant([[[[1, 2], [3, 4]]]], dtype=tf.float32)
    tf.compat.v1.InteractiveSession()  # 开启交互式会话
    print([a.eval(),b.eval()])  # 获取tensor值
    print(a.device,b.device)   # 获取当前tensor运行设备
    # 维度： ()> 0维   (2,)> 1维  (2, 2)>2维  (1, 2, 2)>3维  (1, 1, 2, 2)>4维
    print(a.shape,b.shape,c.shape,d.shape,e.shape)   # 获取数据形状
    print()
    print(a.name)  # 张量名字
    print(a.op)  # 张量字符串描述信息
    print(a.graph)  # 张量所属图地址
    print(tf.is_tensor(a))  # 判断是否为张量
# tensorflow 2.x 版本属性
a = tf.constant([[1, 2, 3], [4, 5, 6]])
print(dir(a))  # device dtype eval shape graph name ndim op numpy
print(a.ndim)  # 获取张量维度
print(a.numpy())  # 将张量转换成 numpy 数组
# todo 总结  op：只要是tensorflow里面api都属于op  张量:op定义出来的数据