# -*- coding: utf-8 -*-

# 在MNIST数据集上训练DAE，保存训练好的模型 DenoisingAutoEncoderRunner.py
# 利用DAE提取的特征来分类（简单的线性分类）DAE_Classification.py
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep
import numpy as np

N_HIDDEN = 200


# 图像归一化
def standard_scale(X_train,X_test):

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# 抽样
def get_random_block(data,label, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)], label[start_index:(start_index+batch_size)]


# 定义分类模型
with tf.name_scope('classification'):
    x = tf.placeholder(tf.float32, shape=[None, N_HIDDEN])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # x_reshape = tf.reshape(x,[-1,20,20,1])
    # tf.summary.image('image',x_reshape,10)

    weight = tf.Variable(tf.zeros(shape=[N_HIDDEN, 10], dtype=tf.float32))
    bias = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))
    logits = tf.add(tf.matmul(x, weight), bias, name='logits')

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cross_entropy_mean)

with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_mean)

with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# 训练步数
STEPS = 3000
batch_size = 100


with tf.Session() as sess:

    # 变量初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 读取保存的模型，并转换成GraphDef Protocol Buffer
    model_filename = 'pb/feature.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 获得定义图的输入(图像)和输出(特征)张量；载入的模型也在默认定义的图中
    jpeg_data_tensor, feature_tensor = tf.import_graph_def(graph_def,
                                                           input_map={'input/noise_scale:0': 0.0},
                                                           return_elements=['input/jpeg_data:0', 'hidden/feature:0'],
                                                           name='feature')

    # 将计算图写入文件
    summary_writer = tf.summary.FileWriter('log', tf.get_default_graph())

    # 获得训练数据并归一化处理
    mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
    train_labels = mnist.train.labels
    test_labels = mnist.test.labels

    # 保存训练图片(183M)和测试图片的特征
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    train_features = sess.run(feature_tensor, feed_dict={jpeg_data_tensor: X_train})
    test_features = sess.run(feature_tensor, feed_dict={jpeg_data_tensor: X_test})

    saver = tf.train.Saver()

    # 开始训练，每隔100步记录summary，每隔200步记录模型
    # 只保留五个最新的model，最保留一个最新的event(包含计算图和加入的变量)
    for i in range(STEPS):
        batch_xs, batch_ys = get_random_block(train_features, train_labels, batch_size)
        loss, _ = sess.run((cross_entropy_mean, train_op), feed_dict={x:batch_xs,y_:batch_ys})

        if i % 100 == 0:
            summary = sess.run(merged,feed_dict={x: batch_xs, y_: batch_ys})
            summary_writer.add_summary(summary, i)
            print('Step %d: loss is %.3f' % (i, loss))

        if (i > 500) and (i % 200 == 0):
            saver.save(sess, 'log/model.ckpt', i)

    summary_writer.close()

    # 测试测试集的准确度
    print(sess.run(accuracy,feed_dict={x: test_features, y_: test_labels}))
