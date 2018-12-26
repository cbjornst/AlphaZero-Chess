# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:19:37 2018

@author: L L L L L
"""

import chess
import numpy as np
from chessPlaying import chessBot
from moveLogic import MCTS
from keras import models
from moveLogic import neuralNet
import random

model1 = neuralNet.chessModel()
model2 = neuralNet.chessModel()
#model1.model = models.load_model('ugh')
#.model2.model = models.load_model('ugh')
hstory = []

def generateData(hstory):
    for j in range(1): 
        games = []
        for i in range(8):
            game = []
            board = chess.Board()
            player1 = chessBot.Player(board, model1)
            player2 = chessBot.Player(board, model2)
            board = chessBot.playChess(player1, player2, board)
            result = board.result()
            if result == '*':
                result = '1/2-1/2'
            game += [result]
            game += [player1.policies]
            while len(board.move_stack) > 0:
                game = [board.pop()] + game
            games += [game]
        print(games)
        games = np.asarray(games)
        np.save('games' + str(j), games)
    trainModel(model1, hstory)

def trainModel(model, hstory):
    for j in range(8):        
        g1 = np.load('games' + str(j) + '.npy')
        miniBatch = []
        values = []
        policies = []
        actual = []
        for i in range(len(g1)):
            board = chess.Board()
            value = g1[i][-2] 
            if value == '1/2-1/2':
                values += [[-1.0]]
            elif value == '1-0':
                values += [[1.0]]
            else:
                values += [[-1.0]]
            policies += [g1[i][-1]]
            moveset = g1[i][:-2]
            if len(moveset) % 2 > 0:
                move = random.randint(0, (len(moveset) - 3)/2)
            else:
                move = random.randint(0, (len(moveset) - 2)/2)  
            for j in range(0,move):
                board.push(moveset[2 * j])
                board.push(moveset[(2 * j) + 1])
            legalMoves = list(board.legal_moves)
            player1 = chessBot.Player(board, model)
            probabilities = model.parseInput(board, 8)
            miniBatch += [probabilities]
            act = np.zeros((73, 8, 8))
            act = player1.tree.getProb(legalMoves, act, board, True, policies[i][move])
            act = np.ndarray.flatten(act)
            actual += [act]
        for i in range(6):
            act2 = np.reshape(actual[i], (1, 4672))
            labels = {'policy_head': act2, 'value_head': np.array(values[i])} 
            history = model.model.fit(miniBatch[i], labels, epochs=1, verbose=1, validation_split=0, batch_size = 1) 
        model.model.save('ugh')
        hstory += [history.history['loss']]
    print(hstory)
    generateData(hstory)       
generateData(hstory)
#trainModel(model1, [])