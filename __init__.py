# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:19:37 2018

@author: L L L L L
"""

import chess
from chessPlaying import chessBot
from moveLogic import MCTS
from moveLogic import neuralNet

board = chess.Board()
model1 = neuralNet.chessModel()
model2 = neuralNet.chessModel()
player1 = chessBot.Player(board, model1)
player2 = chessBot.Player(board, model2)
board = chessBot.playChess(player1, player2, board)

def train():
    return True