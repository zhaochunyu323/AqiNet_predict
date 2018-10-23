#coding:utf-8
import tensorflow as tf
import tensorlayer as tl
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

image_size_x = 12
image_size_y = 12 
batch_size = 128
num_steps = 6
seq_length= 18
output_length = 6
epoches=120
train_size=10944
val_size=2176
test_size=2176
learning_rate=0.001
filename_aqi_x = 'data/all_data_12_x.npy'
filename_aqi_y = 'data/all_data_12_y.npy'
model_path='model_res/model.ckpt'
#model_path_read='/media/zhaochunyu/zhao_model/hebei_predict/model_merge_7_12_0'
model_path_read='model_merge_0_0'

images_test_x_batches_aqi = np.load(filename_aqi_x)[85:]
images_test_y_batches_aqi = np.load(filename_aqi_y)[85:]
#x_aqi = tf.placeholder(tf.float32, shape=[batch_size, num_steps, image_size_x, image_size_y, 1])
x_aqi_short = tf.placeholder(tf.float32, shape=[batch_size, image_size_x, image_size_y, 6])
x_aqi_mid = tf.placeholder(tf.float32, shape=[batch_size, image_size_x, image_size_y, num_steps])
x_aqi_long = tf.placeholder(tf.float32, shape=[batch_size, image_size_x, image_size_y, num_steps])
#y = tf.placeholder(tf.float32, shape=[batch_size, num_steps, image_size_x, image_size_y, 1])
y = tf.placeholder(tf.float32, shape=[batch_size, image_size_x, image_size_y, 6])

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
    net8_aqi_mid = tl.layers.ConcatLayer([net2_aqi_mid, net4_aqi_mid, net5_aqi_mid], -1, name='mergelayer3_mid')
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

with tf.variable_scope('output', reuse = tf.AUTO_REUSE) as scope:
    net_aqi_out = tl.layers.ElementwiseLayer([net3_aqi_long, net3_aqi_mid, net3_aqi_short], combine_fn = tf.add, act = tf.nn.relu, name = 'out_merge' )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_path_read)
    saver.restore(sess, ckpt.model_checkpoint_path)
#    saver.restore(sess,'model_short/model.ckpt-16')
    res_y_pre=[]
    res_in_y=[]

    accuracy_all = 0.0
    total_mae = 0.0
    total_RMSE = 0.0
    station_number = 0
    for x_id in range(0, 12):
        for y_id in range(0, 12):
            list_label = []
            list_gen = []
            station_id_x = x_id
            station_id_y = y_id
            error_ave = 0.0
            p_num = 0.0
            mae = 0.0
            RMSE = 0.0
            if np.load('data/sparse.npy')[x_id, y_id] != 0:
                for num in range(images_test_x_batches_aqi.shape[0]):
                    image_test_x_short=images_test_x_batches_aqi[num,:,12:18,:,:,0]
                    image_test_x_mid=images_test_x_batches_aqi[num,:,6:12,:,:,0]
                    image_test_x_long=images_test_x_batches_aqi[num,:,0:6,:,:,0]
                    image_test_x_short=np.transpose(image_test_x_short,[0,2,3,1])
                    image_test_x_mid=np.transpose(image_test_x_mid,[0,2,3,1])
                    image_test_x_long=np.transpose(image_test_x_long,[0,2,3,1])
                    image_test_y=images_test_y_batches_aqi[num,:,:6,:,:,0]
                    image_test_y=np.transpose(image_test_y,[0,2,3,1])

                    feed_dict={x_aqi_short:image_test_x_short,
                               x_aqi_mid:image_test_x_mid,
                               x_aqi_long:image_test_x_long,
                               y:image_test_y}
                    image_label_all = sess.run(y,feed_dict=feed_dict)
                    image_gen_all = sess.run(net_aqi_out.outputs,feed_dict=feed_dict)
                    error = 0
                    p_batch = 0.0
                    for batch in range(batch_size):
                        error_each=[]
                        numerator = 0.0
                        denominator = 0.0
                        for step in range(6):
                            image_label = image_label_all[batch,:,:,step].reshape(12, 12)*500
                            image_gen = image_gen_all[batch,:,:,step].reshape(12, 12)*500
                            numerator += abs(image_gen[station_id_x,station_id_y]-image_label[station_id_x,station_id_y])
                            mae += abs(image_gen[station_id_x, station_id_y] - image_label[station_id_x, station_id_y])
                            RMSE += pow(abs(image_gen[station_id_x, station_id_y] - image_label[station_id_x, station_id_y]), 2)
                            denominator += image_label[station_id_x,station_id_y]

                        if denominator != 0:
                            p_single = 1 - numerator /denominator
                        else:
                            p_single = 0.8
                        p_batch += p_single
                    p_batch = p_batch/batch_size
                    p_num += p_batch
                p_num = p_num/images_test_x_batches_aqi.shape[0]
            
                accuracy_all += p_num
                station_number += 1
                print 'station ({},{}) accuracy: {}'.format(station_id_x, station_id_y, p_num)
                total_mae += mae/(images_test_x_batches_aqi.shape[0]*batch_size*6)
                total_RMSE += math.sqrt(RMSE / (images_test_x_batches_aqi.shape[0]*batch_size*6))
    print 'total accuracy: {}, mae: {}, RMSE: {}'.format(accuracy_all / station_number, total_mae / station_number, total_RMSE / station_number)
