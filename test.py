#-*-coding:utf8-*-
'''
RNN对数据进行分类
'''

#引入数据
import input_data
mnist = input_data.read_data_sets("/",one_hot=True)

import tensorflow as tf
from tensorflow.models.rnn import rnn,rnn_cell
import numpy as np

#设置参数
learning_rate = 0.001#学习率
training_iters = 100000#迭代次数
batch_size = 128#每次迭代取样个数
display_step = 10 #每个10步进行一次输出

#神经网络参数
n_input = 28
n_steps = 28 #时间窗口
n_hidden = 128 #隐藏层特征
n_classes = 10 #输出的分类种类


#设置输入数据格式
x = tf.placeholder("float",[None,n_steps,n_input])
istate = tf.placeholder("float",[None,2*n_hidden])
y = tf.placeholder("float",[None,n_classes])

#定义权重
weights = {
    'hidden':tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    'hidden':tf.Variable(tf.random_normal([n_hidden])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

def RNN(_X,_istate,_weights,_biases):
    #输入张量
    _X = tf.transpose(_X,[1,0,2])
    #从新调整,输入层到隐藏层
    _X = tf.reshape(_X,[-1,n_input])
    #加入权重
    _X = tf.matmul(_X,_weights['hidden'])+_biases['hidden']

    #定义lSTm神经元细胞,基本的神经元
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0)
    #切割数据
    _X = tf.split(0,n_steps,_X)
    #获得LSTM细胞的输出
    outputs,states = rnn.rnn(lstm_cell,_X,initial_state=_istate)

    #线性激活
    #获得最终的输出
    return tf.matmul(outputs[-1],_weights['out']) + _biases['out']

pred = RNN(x,istate,weights,biases)

#定义损失函数和优化函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))#使用softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#评估模型
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#初始化变量
init = tf.initialize_all_variables()

#载入图
with tf.Session() as sess:
    sess.run(init)
    step = 1
    #一直训练到最大迭代次数
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        #重新切割数据
        batch_xs  = batch_xs.reshape((batch_size,n_steps,n_input))
        #开始训练
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,istate:np.zeros((batch_size,2*n_hidden))})
        if step % display_step == 0:
            #计算精确度
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,istate:np.zeros((batch_size,2*n_hidden))})
            #计算损失函数
            loss = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,istate:np.zeros((batch_size, 2*n_hidden))})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished"
    #测试精确度
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((test_len, 2*n_hidden))})
