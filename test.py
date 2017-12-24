


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
w1=tf.Variable(tf.random_normal([1,2],stddev=1,seed=1))  
    
x=tf.placeholder(tf.float32,shape=(None,2))  
x1=tf.constant([[0.7,0.9]])  
    
a=x+w1  
b=x1+w1  
    
sess=tf.Session()  
sess.run(tf.global_variables_initializer())  

c = numpy.zeros((2,2))
bb = [d.tolist() for d in c]
# b = a.tolist()
y_1=sess.run(a,feed_dict={x:bb})  
y_2=sess.run(b)  
print(y_1)  
print(y_2)  
sess.close  
