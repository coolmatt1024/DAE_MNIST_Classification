# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep
import numpy as np
from DAE_Model import DAE


N_HIDDEN = 200


def standard_scale(X_train, X_test):

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block(data, batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]


# 生成一个自编码器的对象，创建模型
dae = DAE(n_input=784, n_hidden=N_HIDDEN, transfer_function=tf.nn.softplus,
          optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
          scale=0.0)


# 准备训练数据
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 设置训练参数
n_samples = int(mnist.train.num_examples)
training_epochs = 50
batch_size = 100
display_step = 1


for epoch in range(training_epochs):

    total_batch = int(n_samples / batch_size)

    avg_cost = 0.0

    for step in range(total_batch):

        # sample
        x_batch = get_random_block(X_train, batch_size)

        cost = dae.partial_fit(x_batch)
        avg_cost += cost / n_samples

    if epoch % display_step == 0:
        print('Epoch %d,the average cost is %.3f' % (epoch, avg_cost))


print('Total cost:'+str(dae.calc_total_cost(X_test)))

# save the model
dae.save_model_to_pb()


