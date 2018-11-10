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
games = []
for i in range(5):
    game = []
    board = chess.Board()
    player1 = chessBot.Player(board, model1)
    player2 = chessBot.Player(board, model2)
    board = chessBot.playChess(player1, player2, board)
    game += [[board.result()]]
    while len(board.move_stack) > 0:
        game = [board.pop()] + game
    games += [game]
print(games)
games = np.asarray(games)
np.save('games', games)
def train():
    return True