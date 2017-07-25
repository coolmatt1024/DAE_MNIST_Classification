# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import graph_util


# 定义了一个去噪自编码器的类
class DAE(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),
                 scale=0.05):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer_function = transfer_function

        # 训练时加入噪声的比例系数
        self.training_scale = scale

        # 权重系数
        network_weights = self._initial_weights()
        self.weights = network_weights

        # DAE模型
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, n_input], name='jpeg_data')
            self.scale = tf.placeholder(tf.float32, name='noise_scale')

        with tf.name_scope('hidden'):
            # mnist的数据是0-1的浮点数，这里产生一个0-1的正态分布，乘上比例系数作为噪声
            self.hidden = self.transfer_function(tf.add(tf.matmul(self.x + self.scale * tf.random_normal((n_input,)),
                                                                  self.weights['w1']),
                                                 self.weights['b1']), name='feature')

        with tf.name_scope('reconstruction'):
            self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']), self.weights['b2'])

        with tf.name_scope('cost'):
            # cost
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x), 2.0))
            self.optimizer = optimizer.minimize(self.cost)

        # initialize
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # with tf.name_scope('save'):
        #     # save the graph
        #     writer = tf.summary.FileWriter('model/model.ckpt',tf.get_default_graph())
        #     writer.close()

    # 权重初始化，第一层采用了xavier初始化的方式
    def _initial_weights(self):

        all_weights = dict()

        with tf.variable_scope('weights'):
            ## Attention: there exit the xaiver method in layers
            all_weights['w1'] = tf.get_variable('w1', shape=[self.n_input, self.n_hidden],
                                                initializer=tf.contrib.layers.xavier_initializer())
            all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            ## Attention: the usage
            all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input]), dtype=tf.float32)
            all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)

        return all_weights

    # 训练并返回损失
    def partial_fit(self, X):

        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})

        return cost

    # 损失函数也只用于训练
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale
                                                   })

    def transform(self, X):
        return(self.sess.run(self.hidden,
                             feed_dict={self.x: X, self.scale: self.training_scale}))

    def generate(self, hidden=None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden,
                                                             self.scale: self.training_scale})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: X,
                                                             self.scale: self.training_scale})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])

    # 保存图的定义部分，并且将变量常数化
    def save_model_to_pb(self):

        # 得到图的定义部分，并返回一个图的序列化的GraphDef表示
        graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = graph_util.convert_variables_to_constants(self.sess, graph_def, ['hidden/feature'])

        with tf.gfile.GFile('pb/feature.pb', 'wb') as g:
            g.write(output_graph_def.SerializeToString())


