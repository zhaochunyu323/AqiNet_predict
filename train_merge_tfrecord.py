#coding:utf-8
import tensorflow as tf
import tensorlayer as tl
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import weight

image_size_x = 12
image_size_y = 12 
batch_size = 128 
num_steps = 6
seq_length= 18
output_length = 6
epoches=60
train_size=10944
val_size=2176
test_size=2176
learning_rate=0.001
filename_aqi_x = 'data/all_data_12_x.npy'
filename_aqi_y = 'data/all_data_12_y.npy'
filename = 'data/train.tfrecords'
filename_val = 'data/val.tfrecord'


def read_and_decode(filename_):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_)
    features = tf.parse_single_example(serialized_example, features = {
                                      'data_long': tf.FixedLenFeature([864], tf.float32),
                                      'data_mid': tf.FixedLenFeature([864], tf.float32),
                                      'data_short': tf.FixedLenFeature([864], tf.float32),
                                      'data_label': tf.FixedLenFeature([864], tf.float32)})
    data_long = features['data_long']
    data_mid = features['data_mid']
    data_short = features['data_short']
    data_label = features['data_label']
    data_long = tf.reshape(data_long, [6, 12, 12])
#    data_long = tf.transpose(data_long, [1, 2, 0])
    data_mid = tf.reshape(data_mid, [6, 12, 12])
#    data_mid = tf.transpose(data_mid, [1, 2, 0])
    data_short = tf.reshape(data_short, [6, 12, 12])
#    data_short = tf.transpose(data_short, [1, 2, 0])
    data_label = tf.reshape(data_label, [6, 12, 12])
#    data_label = tf.transpose(data_label, [1, 2, 0])
    return data_long, data_mid, data_short, data_label

def inputs(is_train):
    if is_train == True:
        filename_input = filename
    else:
        filename_input = filename_val
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename_input], num_epochs = epoches)
        data_long, data_mid, data_short, data_label = read_and_decode(filename_queue)
        data_batch_long, data_batch_mid, data_batch_short, label_batch = tf.train.shuffle_batch([data_long, data_mid, data_short, data_label],
                                   batch_size = batch_size,
                                  num_threads = 8, capacity = 1000 + 3 * batch_size,
                                  min_after_dequeue = 1000, enqueue_many = False)
#        data_batch_long, data_batch_mid, data_batch_short, label_batch = tf.train.shuffle_batch([data_long, data_mid, data_short, data_label],
#                                   batch_size = batch_size,
#                                  num_threads = 8, capacity = 1000 + 3*128, min_after_dequeue = 1000,
#                                  enqueue_many = False)
    return data_batch_long, data_batch_mid, data_batch_short, label_batch

def inference(x_aqi_long, x_aqi_mid, x_aqi_short):
    with tf.variable_scope('long', reuse = tf.AUTO_REUSE) as scope:
        net1_aqi_long = tl.layers.InputLayer(x_aqi_long, name='inputlayer1_long')
        net2_aqi_long = tl.layers.Conv2d(net1_aqi_long, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn1_long')
#    net9_aqi_long = tl.layers.Conv2d(net1_aqi_long, n_filter=128, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn5_long')
#    net10_aqi_long = tl.layers.ElementwiseLayer([net2_aqi_long, net9_aqi_long], combine_fn=tf.add, act=tf.nn.relu, name='mergelayer4_long')
        net4_aqi_long = tl.layers.Conv2d(net2_aqi_long, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn3_long')
        net7_aqi_long = tl.layers.ConcatLayer([net2_aqi_long, net4_aqi_long], -1, name='mergelayer2_long')
        net5_aqi_long = tl.layers.Conv2d(net7_aqi_long, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn4_long')
        net8_aqi_long = tl.layers.ConcatLayer([net2_aqi_long, net4_aqi_long, net5_aqi_long], -1, name='mergelayer3_long')
        net3_aqi_long = tl.layers.Conv2d(net8_aqi_long, n_filter=6, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn2_long')               

    with tf.variable_scope('mid', reuse = tf.AUTO_REUSE) as scope:
        net1_aqi_mid = tl.layers.InputLayer(x_aqi_mid, name='inputlayer1_mid')
        net2_aqi_mid = tl.layers.Conv2d(net1_aqi_mid, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn1_mid')
#    net9_aqi_mid = tl.layers.Conv2d(net1_aqi_mid, n_filter=128, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn5_mid')
#    net10_aqi_mid = tl.layers.ElementwiseLayer([net2_aqi_mid, net9_aqi_mid], combine_fn=tf.add, act=tf.nn.relu, name='mergelayer4_mid')
        net4_aqi_mid = tl.layers.Conv2d(net2_aqi_mid, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn3_mid')
        net7_aqi_mid = tl.layers.ConcatLayer([net2_aqi_mid, net4_aqi_mid], -1, name='mergelayer2_mid')
        net5_aqi_mid = tl.layers.Conv2d(net7_aqi_mid, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn4_mid')
        net8_aqi_mid = tl.layers.ConcatLayer([net2_aqi_mid, net4_aqi_mid, net5_aqi_mid], -1, name='mergelayer3_mid')    #attention: add a layer from long term
        net3_aqi_mid = tl.layers.Conv2d(net8_aqi_mid, n_filter=6, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn2_mid')               
 
    with tf.variable_scope('short', reuse = tf.AUTO_REUSE) as scope:
        net1_aqi_short = tl.layers.InputLayer(x_aqi_short, name='inputlayer1_short')
        net2_aqi_short = tl.layers.Conv2d(net1_aqi_short, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn1_short')
#    net9_aqi = tl.layers.Conv2d(net1_aqi, n_filter=128, filter_size=(5, 5), strides=(1, 1),
#            act=tf.nn.relu, padding='SAME', name='cnn5_short')
#    net10_aqi = tl.layers.ElementwiseLayer([net2_aqi, net9_aqi], combine_fn=tf.add, act=tf.nn.relu, name='mergelayer4_short')
        net4_aqi_short = tl.layers.Conv2d(net2_aqi_short, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn3_short')
        net7_aqi_short = tl.layers.ConcatLayer([net2_aqi_short, net4_aqi_short], -1, name='mergelayer2_short')
        net5_aqi_short = tl.layers.Conv2d(net7_aqi_short, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn4_short')
        net8_aqi_short = tl.layers.ConcatLayer([net2_aqi_short, net4_aqi_short, net5_aqi_short], -1, name='mergelayer3_short')
        net3_aqi_short = tl.layers.Conv2d(net8_aqi_short, n_filter=6, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='cnn2_short')               

    with tf.variable_scope('out', reuse = tf.AUTO_REUSE) as scope:
        net_aqi_out = tl.layers.ElementwiseLayer([net3_aqi_long, net3_aqi_mid, net3_aqi_short], combine_fn = tf.add, act = tf.nn.relu, name = 'outlayer')
#    with tf.variable_scope('merge', reuse = tf.AUTO_REUSE) as scope:
#        initializer = tf.constant_initializer(value = [1 / 144.], dtype = tf.float32)
#        w1 = tf.get_variable('short', [128, 12, 12, 6], dtype = tf.float32, initializer = initializer)
#        w2 = tf.get_variable('mid', [128, 12, 12, 6], dtype = tf.float32, initializer = initializer)
#        w3 = tf.get_variable('long', [128, 12, 12, 6], dtype = tf.float32, initializer = initializer)
#        net_out = tf.multiply(net3_aqi_short.outputs, w1) + tf.multiply(net3_aqi_mid.outputs, w2) + tf.multiply(net3_aqi_long.outputs, w3)
    return net_aqi_out.outputs

def loss(out, label):
    cost = tf.reduce_mean(tf.reduce_mean(tf.abs(tf.subtract(out, label)), -1), 0)
    return cost 

def train():
    global kesi
    kesi = 0
    alpha_all = []
    for i, name in enumerate(['out']):
        model_path = 'model_merge_%d_/model.ckpt'%i
        model_path_read = 'model_merge_%d_0'%i
        if i == 0:
            w = np.zeros(12 * 12)   #第一次赋予样本的权重为1/144
            for x in range(w.shape[0]):
                w[x] = 1 / float(w.shape[0])
            print w
            w_before = w
        w = w.reshape(12, 12)
        w_tensor = tf.convert_to_tensor(w, tf.float32)
        data_long, data_mid, data_short, data_label = inputs(is_train = True)
        data_long = tf.transpose(data_long, [0, 2, 3, 1])
        data_mid = tf.transpose(data_mid, [0, 2, 3, 1])
        data_short = tf.transpose(data_short, [0, 2, 3, 1])
        data_label = tf.transpose(data_label, [0, 2, 3, 1])
        prediction = inference(data_long, data_mid, data_short)
        losses = loss(prediction, data_label)
        losses = tf.reduce_mean(tf.multiply(losses, w_tensor))
        with tf.variable_scope('learning_rate'):
            lr = tf.Variable(learning_rate, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(losses, tvars), 5)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(lr)
        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE) as scope:
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        saver=tf.train.Saver(max_to_keep=5)
        with tf.Session() as sess:
            sess.run(init_op)
            tf.train.start_queue_runners(sess = sess)
            tl.layers.initialize_global_variables(sess)
#            ckpt = tf.train.get_checkpoint_state(model_path_read)
#            saver.restore(sess, ckpt.model_checkpoint_path)
            step = 0
            total_cost = 0
            while step < 5100:
                _loss, _ = sess.run([losses, train_op])
                total_cost += _loss
                if step % 85 == 0 and step != 0:
                    print 'epoch %d train loss: %f'%(step / 85, total_cost / 85.)
                    total_cost = 0
                    saver.save(sess, model_path, global_step = step)
                step += 1 
        error_all = weight.error(model_path_read, name)
        if i == 0:
            kesi_1, alpha_1 = weight.kesi_alpha(w_before, error_all)
            alpha_all.append(alpha_1)
            kesi = kesi_1
            print alpha_1
            w = weight.weight(error_all, kesi, w_before)
            w_before = w
            print w
            print kesi
        else:
            kesi, alpha = weight.kesi_alpha(w_before, error_all)
            alpha_all.append(alpha)
            w = weight.weight(error_all, kesi, w_before)
            w_before = w
            print alpha
            print w
            print kesi
    return alpha_all 

def main(argv = None):
    alpha = train()
    print alpha
if __name__ == '__main__':
    tf.app.run()
