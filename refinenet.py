
import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import Variable
from chainer import initializers

from resnet101 import Block

class MRF(chainer.Chain):
  def  __init__(self, filters):
    super(MRF, self).__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(filters, filters, 3, 1, 1)
      self.conv2 = L.Convolution2D(filters, filters, 3, 1, 1)
      self.upsample = L.Deconvolution2D(filters, filters, ksize=4, stride=2, pad=1)
  def __call__(self, x, y=None):
    h1 = self.conv1(x)

    if y is None:
      return h1
    else:
      h2 = self.upsample(self.conv2(y))
      return h1+h2

class RCU(chainer.Chain):
  def __init__(self, filters):
    super(RCU, self).__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(filters, filters, 3, 1, 1)
      self.conv2 = L.Convolution2D(filters, filters, 3, 1, 1)

  def __call__(self, x):
    h = F.relu(x)
    h = F.relu(self.conv1(h))
    h = self.conv2(h)
    return x+h

class CRP(chainer.Chain):
  def __init__(self, filters):
    super(CRP, self).__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(filters, filters, 3, 1, 1)
      self.conv2 = L.Convolution2D(filters, filters, 3, 1, 1)
  def __call__(self, x):
    h = F.relu(x)
    c = F.max_pooling_2d(h, 5, stride=1, pad=2)
    c = self.conv1(c)
    h = h+c
    c = F.max_pooling_2d(c, 5, stride=1, pad=2)
    c = self.conv2(c)
    return h+c

class RefineNet(chainer.Chain):
  def __init__(self, filters):
    super(RefineNet, self).__init__()
    with self.init_scope():
      self.rcu1_1 = RCU(filters)
      self.rcu1_2 = RCU(filters)
      self.rcu2_1 = RCU(filters)
      self.rcu2_2 = RCU(filters)

      self.mrf = MRF(filters)
      self.crp = CRP(filters)
      self.rcu3 = RCU(filters)
    
  def __call__(self, x0, y0=None):
    x1 = self.rcu1_2(self.rcu1_1(x0))
    if y0 is None:
      h = self.mrf(x1)
    else:
      y1 = self.rcu2_2(self.rcu2_1(y0))
      h = self.mrf(x1, y1)

    return self.rcu3(self.crp(h))

class RefineResNet(chainer.Chain):

  def __init__(self, class_num):
    super(RefineResNet, self).__init__()
    with self.init_scope():
      self.conv1 = L.Convolution2D(
        3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
      self.bn1 = L.BatchNormalization(64)
      self.res2 = Block(3, 64, 64, 256, 1)
      self.res3 = Block(4, 256, 128, 512)
      self.res4 = Block(23, 512, 256, 1024)
      self.res5 = Block(3, 1024, 512, 2048)

      fn0 = 256
      fn1 = 512
      
      self.pool2=L.Convolution2D(256, fn0, 1, stride=1, pad=0)
      self.pool3=L.Convolution2D(512, fn0, 1, stride=1, pad=0)
      self.pool4=L.Convolution2D(1024, fn0, 1, stride=1, pad=0)
      self.pool5=L.Convolution2D(2048, fn1, 1, stride=1, pad=0)
      
      self.upsample = L.Deconvolution2D(fn0, fn0, ksize=8, stride=4, pad=2)
      self.rfn2 = RefineNet(fn0) # 1/4  RefineNet1
      self.rfn3 = RefineNet(fn0) # 1/8  RefineNet2
      self.rfn4 = RefineNet(fn0) # 1/16 RefineNet3
      self.rfn5 = RefineNet(fn1) # 1/32 RefineNet4

      self.pool6=L.Convolution2D(fn1, fn0, 1, stride=1, pad=0)
      
      self.rcu1 = RCU(fn0)
      self.rcu2 = RCU(fn0)

      self.final =L.Convolution2D(fn0, class_num, 1, stride=1, pad=0)

  def __call__(self, x, t=None, train=False, test=False):    
    h = self.bn1(self.conv1(x))
    h = F.max_pooling_2d(F.relu(h), 3, stride=2) # 1/2
    h2 = self.res2(h)        # 1/4
    h3 = self.res3(h2)       # 1/8
    h4 = self.res4(h3)       # 1/16
    h5 = self.res5(h4)       # 1/32

    c2 = self.pool2(h2)
    c3 = self.pool3(h3)
    c4 = self.pool4(h4)
    c5 = self.pool5(h5)

    r5 = self.pool6(self.rfn5(c5))
    r4 = self.rfn4(c4, r5)
    r3 = self.rfn3(c3, r4)
    r2 = self.rfn2(c2, r3)

    o = self.rcu1(r2)
    o = self.rcu1(o)    
    
    h = self.upsample(o)
    h = self.final(h)

    if chainer.config.train:
      loss = F.softmax_cross_entropy(h, t)
      return loss
    else:
      pred = F.softmax(h)
      return pred
