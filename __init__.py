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
board = chessBot.playChess(board)
model = neuralNet.chessModel()
model.parseInput(board, 8)
tree = MCTS.MCST(board, 1, 0, 1)
node = tree.nextNode(board, 0, 1)

def train():
    return True