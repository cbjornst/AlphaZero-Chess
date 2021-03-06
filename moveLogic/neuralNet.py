# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:17:04 2018

@author: L L L L L
"""
import os
from keras.layers import Conv2D, Flatten, Input, Dense, BatchNormalization, ReLU, Add
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras import losses
import numpy as np
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

pieceValues = {'p': 0, 'b': 1, 'n': 2, 'r': 3, 'q': 4, 'k': 5, 'P': 6, 'B': 7,
                   'N': 8, 'R': 9, 'Q': 10, 'K': 11}

class chessModel:
    def __init__(self):
        self.model = None
        self.buildModel()
        
    def parseOneInput(self, board, T, result, move):
        #called for each timestep T
        newBoard = np.chararray([8, 8], unicode=True)
        pm = board.piece_map()
        for i in board.piece_map():
            newBoard[i // 8][i % 8] = pm[i].symbol()
        for i in range(8):
            for j in range(8):
                if newBoard[i][j] is not '':
                    layer = pieceValues[newBoard[i][j]] + (12 * T)
                    result[layer][i][j] = 1.0        
                
    def parseInput(self, board, T):
        #parse a board state and convert it into a Neural Net input
        result = np.zeros((119, 8, 8))
        np.set_printoptions(threshold=np.inf)
        board2 = board.copy()
        for i in range(T):
            if len(board2.move_stack) > 0:
                move = board2.peek()
            else:
                move = None
            self.parseOneInput(board2, i, result, move)
            if len(board2.move_stack) > 0:
                board2.pop()
            else: 
                break
        if board.turn:
            result[96] += 1
        else:
            result[96] += 2
        result[97] += board.fullmove_number
        if board.has_kingside_castling_rights(0):
            result[98] += 1
        if board.has_kingside_castling_rights(1):
            result[99] += 1
        if board.has_queenside_castling_rights(0):
            result[100] += 1
        if board.has_queenside_castling_rights(1):
            result[101] += 1
        if board.can_claim_fifty_moves():
            result[102] += 1
        return result.reshape(1, 119, 8, 8)            
    
    def residualLayer(self, x):
        resBlock = Conv2D(76, kernel_size=4, padding='same', strides=1)(x)
        resBlock = BatchNormalization()(resBlock)
        resBlock = ReLU()(resBlock)
        resBlock = Conv2D(76, kernel_size=4, padding='same', strides=1)(resBlock)
        resBlock = BatchNormalization()(resBlock)
        resBlock = Add()([x, resBlock])
        resBlock = ReLU()(resBlock)
        return resBlock
        
    def policyHead(self, x):
        policy = Conv2D(2, kernel_size=1, padding='same', strides=1)(x)
        policy = BatchNormalization()(policy)
        policy = ReLU()(policy)
        policy = Flatten()(policy)
        policy = Dense(4672, name = 'policy_head')(policy)
        return policy
        
    def valueHead(self, x):
        value = Conv2D(1, kernel_size=1, padding='same', strides=1)(x)
        value = BatchNormalization()(value)
        value = ReLU()(value)
        value = Flatten()(value)
        value = Dense(256)(value)
        value = ReLU()(value)
        value = Dense(1, activation='tanh', name = 'value_head')(value)
        return value
        
    def lossFunction(self, y_true, y_pred):
        loss1 = losses.mean_squared_error(y_true, y_pred)
        loss2 = losses.categorical_crossentropy(y_true, y_pred)
        return loss1 + loss2
        
    def buildModel(self):
        #compiles the neural net
        inputStack = Input(shape=(119, 8, 8))
        x = Conv2D(76, kernel_size=4, strides=1, padding='same', input_shape=(119, 8, 8))(inputStack)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        for i in range(2):
            x = self.residualLayer(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        losses = {"value_head": "mean_squared_error", "policy_head": "categorical_crossentropy"}
        lossWeights = {"policy_head": .5, "value_head": .5}
        self.model = Model(inputs=inputStack, outputs=(policy, value))
        self.model.compile(optimizer=RMSprop(lr=.03), loss=losses, metrics=['accuracy'], loss_weights=lossWeights)
        #self.model.compile(optimizer='rmsprop', loss=self.lossFunction, metrics=['accuracy'])
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #SVG(model_to_dot(self.model).create(prog='dot', format='svg'))
        #plot_model(self.model, to_file='model.png')
            
    
    def runModel(self, node):
        #gets predictions
        probs, v = self.model.predict_on_batch(node)
        return probs, v