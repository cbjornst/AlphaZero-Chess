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
model1.model = models.load_model('ugh')
model2 = neuralNet.chessModel()

def generateData():
    for j in range(1): 
        games = []
        for i in range(4):
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
        games = np.asarray(games)
        np.save('games' + str(j), games)
        trainModel(model1)

def trainModel(model):
    g1 = np.load('games0.npy')
    miniBatch = []
    values = []
    policies = []
    actual = []
    for i in range(len(g1)):
        board = chess.Board()
        value = g1[i][-2] 
        if value == '1/2-1/2':
            values += [[0]]
        elif value == '1-0':
            values += [[1]]
        else:
            value += [0]
        policies += [g1[i][-1]]
        moveset = g1[i][:-2]
        if len(moveset) % 2 > 0:
            move = random.randint(8, 21)
        else:
            move = random.randint(8, 21)  
        for j in range(0,move):
            board.push(moveset[2 * j])
            board.push(moveset[(2 * j) + 1])
        legalMoves = list(board.legal_moves)
        player1 = chessBot.Player(board, model)
        probabilities = model1.parseInput(board, 8)
        miniBatch += [probabilities]
        act = np.zeros((73, 8, 8))
        act = player1.tree.getProb(legalMoves, act, board, True, policies[i][move])
        act = np.ndarray.flatten(act)
        actual += [act]
    for i in range(4):
        act2 = np.reshape(actual[i], (1, 4672))
        labels = {'policy_head': act2, 'value_head': np.array(values[i])} 
        history = model.model.fit(miniBatch[i], labels, epochs=2, verbose=1, validation_split=0, batch_size = 1) 
    model.model.save('ugh')
    generateData()
    print(values)
    print(history.history['loss'])       
generateData()
#trainModel(model1)