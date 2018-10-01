# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:17:04 2018

@author: L L L L L
"""

from keras.layers import Conv2D, Flatten, Input, Dense, BatchNormalization, ReLU, Add
from keras.models import Model
from keras import losses
import numpy as np

pieceValues = {'p': 0, 'b': 1, 'n': 2, 'r': 3, 'q': 4, 'k': 5, 'P': 6, 'B': 7,
                   'N': 8, 'R': 9, 'Q': 10, 'K': 11}
result = np.zeros((119, 8, 8))

def parseOneInput(board, T):
    newBoard = np.chararray([8, 8], unicode=True)
    pm = board.piece_map()
    for i in board.piece_map():
        newBoard[i // 8][i % 8] = pm[i].symbol()
    for i in range(8):
        for j in range(8):
            if newBoard[i][j] is not '':
                layer = pieceValues[newBoard[i][j]] + (12 * T)
                result[layer][i][j] = 1.0

def parseInput(board, T):
    np.set_printoptions(threshold=np.inf)
    board2 = board.copy()
    for i in range(T):
        parseOneInput(board2, T)
        board2.pop()
    return result.reshape(119, 8, 8, 1)            

def residualLayer(x):
    resBlock = Conv2D(256, kernel_size=3, padding='same', strides=1)(x)
    resBlock = BatchNormalization()(resBlock)
    resBlock = ReLU()(resBlock)
    resBlock = Conv2D(256, kernel_size=3, padding='same', strides=1)(resBlock)
    resBlock = BatchNormalization()(resBlock)
    resBlock = Add()([x, resBlock])
    resBlock = ReLU()(resBlock)
    return resBlock
    
def policyHead(x):
    policy = Conv2D(2, kernel_size=1, strides=1)(x)
    policy = BatchNormalization()(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    policy = Dense(4672)(policy)
    return policy
    
def valueHead(x):
    value = Conv2D(1, kernel_size=1, strides=1)(x)
    value = BatchNormalization()(value)
    value = ReLU()(value)
    return value
    
def lossFunction(y_true, y_pred):
    loss1 = losses.mean_squared_error(y_true, y_pred)
    loss2 = losses.categorical_crossentropy(y_true, y_pred)
    return loss1 + loss2
    
def runModel(node):
    inputStack = Input(batch_shape=(119, 8, 8, 1))
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', input_shape=(119, 8, 8, 1))(inputStack)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    for i in range(39):
        x = residualLayer(x)
    policy = policyHead(x)
    value = valueHead(x)
    
    model = Model(inputs=inputStack, outputs=[policy, value])
    model.compile(optimizer='rmsprop', loss=lossFunction, metrics=['accuracy'])
    dumb, v = model.predict(node)
    return dumb, v