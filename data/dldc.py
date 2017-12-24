
"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
# import gzip

import numpy
# from six.moves import xrange  # pylint: disable=redefined-builtin

# from tensorflow.contrib.learn.python.learn.datasets import base
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import random_seed
# from tensorflow.python.platform import gfile


import tensorflow as tf
#for excel
from openpyxl import load_workbook

# read Data
Data_dir = '/home/yangxianping/share/孳生地预测/'

Train_data_with_files = ('with_20150414', 'with_20151020', 'with_20150414_2', 'with_20151021')
Train_data_without_file = ('without',)

Train_data_with = []
Train_data_without = []


for file in Train_data_with_files:
    try:
        tmp = numpy.loadtxt(Data_dir + file + '.txt')
        print(tmp.shape)
        Train_data_with.append(tmp)
    except Exception as e:
        print('failed to open file ' + Data_dir + file + '.txt')
        print(e)

for file in Train_data_without_file:
    try:
        tmp = numpy.loadtxt(Data_dir + file + '.txt')
        print(tmp.shape)
        Train_data_without.append(tmp)
    except Exception as e:
        print('failed to open file ' + file + '.txt')
        print(e)

print('data with shape:' + str(len(Train_data_with)))
print('data without shape:' + str(len(Train_data_without)))

origin_x1 = []
origin_x0 = []
density = []
for d in Train_data_with:
    for dd in d:
        mask = dd > 0
        if False not in mask:
            origin_x1.append(dd[:5])
            density.append(dd[5])

print('origin_x size:'+ str(len(origin_x1)))
# Train_data_with = numpy.array(origin_x1)
# print('Train_data_with size:'+ str(Train_data_with.shape))

for d in Train_data_without:
    for dd in d:
        mask = dd > 0
        if False not in mask:
            origin_x0.append(dd)

print('origin_y size:'+ str(len(origin_x0)))
# Train_data_without = numpy.array(origin_x0)
# print('Train_data_without size:'+ str(Train_data_without.shape))

# create the training data
Train_data = []
Label = []
Check_data = []
Check_Label = []

Check_data = origin_x1[:120]
Train_data = origin_x1[120:]

Check_data[120:] = origin_x0[:5000]
Train_data[len(origin_x1[120:]):] = origin_x0[5000:]

Train_data = numpy.array(Train_data)

Label = numpy.zeros((len(origin_x1[120:]) + len(origin_x0[5000:]), 2))
for a in range(len(origin_x1[120:])):
    Label[a,1] = 1

for a in range(len(origin_x1[120:]), len(origin_x1[120:]) + len(origin_x0[5000:])):
    Label[a,0] = 1

Check_Label = numpy.zeros((len(origin_x1[:120]) + len(origin_x0[:5000]), 2))
for a in range(len(origin_x1[:120])):
    Check_Label[a,1] = 1

for a in range(len(origin_x1[:120]), len(origin_x1[:120]) + len(origin_x0[:5000])):
    Check_Label[a,0] = 1

# print(Label)
Label = Label.tolist()
Check_Label = Check_Label.tolist()

print('Check_data:shape'+str(len(Check_data)))
print('Train_data:shape'+str(len(Train_data)))
print('Label:shape'+str(len(Label)))
print('Check_Label:shape'+str(len(Check_Label)))

# Create the model
x = tf.placeholder(tf.float32, [None, 5])
W = tf.Variable(tf.zeros([5, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(2000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: Train_data, y_: Label})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: Check_data,
                                    y_: Check_Label}))
