
"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
# import gzip

import numpy
import random
# from six.moves import xrange  # pylint: disable=redefined-builtin

# from tensorflow.contrib.learn.python.learn.datasets import base
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import random_seed
# from tensorflow.python.platform import gfile


import tensorflow as tf
#for excel
from openpyxl import load_workbook

# read Data
Data_dir = '/home/super/hhd/dldc/data/'

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
            density.append(numpy.array(dd[5]))

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
check_num = 200
check_num_without = 8000

Check_data = origin_x1[:check_num]
Train_data = origin_x1[check_num:]

Check_data[check_num:] = origin_x0[:check_num_without]
Train_data[len(origin_x1[check_num:]):] = origin_x0[check_num_without:]

# Train_data = numpy.array(Train_data)

Label = numpy.zeros((len(origin_x1[check_num:]) + len(origin_x0[check_num_without:]), 2))
for a in range(0, len(origin_x1[check_num:])):
    Label[a,1] = 1

for a in range(len(origin_x1[check_num:]), len(origin_x1[check_num:]) + len(origin_x0[check_num_without:])):
    Label[a,0] = 1

# error = []
# for a in Label:
#     if a[0] == 0 and a[1] == 0:
#         error.append(a)

Check_Label = numpy.zeros((len(origin_x1[:check_num]) + len(origin_x0[:check_num_without]), 2))
for a in range(len(origin_x1[:check_num])):
    Check_Label[a,1] = 1

for a in range(len(origin_x1[:check_num]), len(origin_x1[:check_num]) + len(origin_x0[:check_num_without])):
    Check_Label[a,0] = 1

# error = []
# for a in Label:
#     if a[0] == 0 and a[1] == 0:
#         error.append(a)

# print(Label)
Label = Label.tolist()
Check_Label = Check_Label.tolist()

Label = [numpy.array(dd) for dd in Label]

print('Check_data:shape'+str(len(Check_data)))
print('Train_data:shape'+str(len(Train_data)))
print('Train_data type' + str(type(Train_data)) + '  data type ' + str(type(Train_data[1])))
print('Label type' + str(type(Label)) + '  data type ' + str(type(Label[1])))

print('Label:shape'+str(len(Label)))
print('Check_Label:shape'+str(len(Check_Label)))

# Create the model
x = tf.placeholder(tf.float32, [None, 5])
W = tf.Variable(tf.zeros([5, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

indexrange = range(0, len(Train_data))
count_one = int(len(Train_data) / 10)

resultindex = random.sample(indexrange, count_one)



# Train
for _ in range(10):
    batch_xs = []
    batch_ys = []
    for dd in resultindex:
        batch_xs.append(numpy.array(Train_data[dd]))
        batch_ys.append(numpy.array((Label[dd])))
    # print('batch_xs_array:shape: '+str(len(batch_xs)))
    # print('batch_ys_array:shape: '+str(len(batch_ys)))

    # print('batch_xs type' + str(type(batch_xs)) + '  data type ' + str(type(batch_xs[1])))
    # print('batch_ys type' + str(type(batch_ys)) + '  data type ' + str(type(batch_ys[1])))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # sess.run(train_step, feed_dict={x: Train_data, y_: Label})

print(W)
print(b)
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: Check_data,
                                    y_: Check_Label}))


# step 2 : train the density prediction

#origin_x1 density
    
# density = [list(dd) for dd in density]

Density_check_num = 200
Density_Train_x = origin_x1[Density_check_num:]
Density_Train_y = density[Density_check_num:]

Density_Check_x = origin_x1[:Density_check_num]
Density_Check_y = density[:Density_check_num]


print('Density_Train_y:shape'+str(len(Density_Train_y)))
print('Density_Train_x:shape'+str(len(Density_Train_x)))
print('Density_Train_x type' + str(type(Density_Train_x)) + '  data type ' + str(type(Density_Train_x[1])))
print('Density_Check_y type' + str(type(Density_Train_y)) + '  data type ' + str(type(Density_Train_y[1])))

# Create the model
xx = tf.placeholder(tf.float32, [None, 5])
WW = tf.Variable(tf.zeros([5, 1]))
bb = tf.Variable(tf.zeros([1]))
yy = tf.nn.softmax(tf.matmul(xx, WW) + bb)

# Define loss and optimizer
yy_ = tf.placeholder(tf.float32, [None])

cross_entropy2 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=yy_, logits=yy))
train_step2 = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy2)

sess2 = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(10):
    sess.run(train_step2, feed_dict={xx: Density_Train_x, yy_: Density_Train_y})

print(WW)
print(bb)
# Test trained model
# correct_prediction2 = tf.div(tf.sub(yy, yy_), yy_)
accuracy2 = tf.reduce_mean(tf.squared_difference(yy, yy_))
print(sess.run(accuracy2, feed_dict={xx: Density_Check_x,
                                    yy_: Density_Check_y}))