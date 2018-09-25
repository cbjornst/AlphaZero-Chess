# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:19:37 2018

@author: L L L L L
"""

import chess
from chessPlaying import chessBot
from moveLogic import MCTS

board = chess.Board()
chessBot.playChess()
print(MCTS.move(board))