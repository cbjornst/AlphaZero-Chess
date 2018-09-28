# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:17:04 2018

@author: L L L L L
"""

from keras.layers import Conv2D, Flatten, Input, Dense, BatchNormalization, ReLU, Add
from keras.models import Model
from keras import losses

def residualLayer(x):
    resBlock = Conv2D(256, kernel_size=(3,3), strides=(1,1))(x)
    resBlock = BatchNormalization()(resBlock)
    resBlock = ReLU()(resBlock)
    resBlock = Conv2D(256, kernel_size=(3,3), strides=(1,1))(resBlock)
    resBlock = BatchNormalization()(resBlock)
    resBlock = Add()([resBlock, x])
    resBlock = ReLU()(resBlock)
    
def policyHead(x):
    policy = Conv2D(2, kernel_size=(1,1), strides=(1,1))(x)
    policy = BatchNormalization()(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    policy = Dense(4,672)(policy)
    
def valueHead(x):
    value = Conv2D(1, kernel_size=(1,1), strides=(1,1))(x)
    value = BatchNormalization()(value)
    value = ReLU()(value)
    
def lossFunction(y_true, y_pred):
    loss1 = losses.mean_squared_error(y_true, y_pred)
    loss2 = losses.categorical_crossentropy(y_true, y_pred)
    return loss1 + loss2
    
def runModel(node):
    inputStack = Input(shape=(8, 8, 119))
    x = Conv2D(kernel_size=256, strides=(1, 1))(inputStack)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    for i in range(39):
        x = residualLayer(x)
    policy = policyHead(x)
    value = valueHead(x)
    
    model = Model(inputs=[inputStack], outputs=[policy, value])
    model.compile(optimiser='rmsprop', loss=lossFunction, metrics=['accuracy'])