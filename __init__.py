# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:19:37 2018

@author: L L L L L
"""

import chess
import numpy as np
from chessPlaying import chessBot
from moveLogic import MCTS
from moveLogic import neuralNet

model1 = neuralNet.chessModel()
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
            game += [[board.result(), player1.policies]]
            while len(board.move_stack) > 0:
                game = [board.pop()] + game
            games += [game]
        games = np.asarray(games)
        np.save('games' + str(j), games)

def trainModel(model):
    g1 = np.load('games1.npy')
    print(g1)
    g2 = np.load('games2.npy')
    g3 = np.load('games3.npy')
    g4 = np.load('games0.npy')
    
generateData()
#trainModel(model1)