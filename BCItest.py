import tensorflow as tf
import scipy.io as scio
import pandas as pd
import numpy as np

class DataSet(object):
 
    def __init__(self, images, labels, num_examples):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0   # 调用next_batch()函数后记住上一次位置
        self._num_examples = num_examples  # 训练样本数
 
    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epochs
 
        if self._epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self._num_examples)
            np.random.shuffle(index0)
            self._images = np.array(self._images)[index0]
            self._labels = np.array(self._labels)[index0]
 
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                index = np.arange(self._num_examples)
                np.random.shuffle(index)
                self._images = self._images[index]
                self._labels = self._labels[index]
            start = 0
            self._index_in_epochs = batch_size - rest_num_examples
            end = self._index_in_epochs
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
 
        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self._images[start:end], self._labels[start:end]


data_path="dataset_BCIcomp1.mat"
data_train = scio.loadmat(data_path)

orgin_data_train_data = data_train.get('x_train')
new_data_train_data = orgin_data_train_data[3*128:9*128]
data_train_data = np.reshape(new_data_train_data,(-1,140))
data_train_data = data_train_data[:,-10:].transpose()


orgin_data_train_label = data_train.get('y_train')
data_train_label=np.zeros((140,2))
for pos in range(len(orgin_data_train_label)):
    label = orgin_data_train_label[pos]
    if label == [1]:
        data_train_label[pos]=[1,0]
    else:
        data_train_label[pos]=[0,1]
data_train_label = data_train_label[-10:,:]


ds = DataSet(data_train_data,data_train_label,10)

def weight_variable(shape):  #用对称破坏的小噪声初始化权重
    initial =  tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):  #将偏置参数初始化为小的正数，以避免死神经元
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x,ksize=[1,1,4,1],strides=[1,1,4,1],padding='SAME')

x = tf.placeholder(tf.float32, shape=[None,2304],name='x')
# 初始化输出Y
y_ = tf.placeholder(tf.float32,shape=[None,2],name='y_')

#第一个卷积层
W_conv1 =  weight_variable([3,26,3,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,6,128,3])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#h_pool1的输出即为第一层网络输出，shape为[batch,9,32,32]
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层
W_conv2 = weight_variable([3,26,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
#该层拥有1024个神经元
#W的第1维size为9*8*64，9*8是h_pool2输出的size，64是第2层输出神经元个数
w_fc1 = weight_variable([6 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
# 将第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

#Dropout减少过拟合
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


#读出层
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)
    )

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()

#无接口运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "D:/CSU/program/Python/Practice/Brain_cognition/Brain_cognition/model.ckpt") #使用模型，参数和之前的代码保持一致
    batch = ds.next_batch(5)
    test_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 }, session=sess)
    print('test accuracy %g' % test_accuracy)







#接口运行
#with tf.Session() as sess:

#    sess.run(tf.global_variables_initializer())
#    saver = tf.train.import_meta_graph('D:/CSU/program/Python/Practice/Brain_cognition/Brain_cognition/model.ckpt.meta')
#    saver.restore(sess, 'D:/CSU/program/Python/Practice/Brain_cognition/Brain_cognition/model.ckpt')
#    pred = tf.get_collection('network-output')[0]
#    graph = tf.get_default_graph()

#    x = graph.get_operation_by_name('x').outputs[0]
#    y_ = graph.get_operation_by_name('y_').outputs[0]
#    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
#    #out = graph.get_tensor_by_name('prediction/output:0')

#    print('识别结果：')
#    print(sess.run(pred, feed_dict={x:data_train_data, y_:data_train_label, keep_prob:0.5}))
