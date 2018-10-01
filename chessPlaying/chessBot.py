# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:28:33 2018

@author: L L L L L
"""

import chess
import random
from moveLogic import neuralNet

def philPlayer(moves):
    return moves[0]

def gregPlayer(moves):
    return random.choice(moves)

def gameOverReason(board):
    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.is_insufficient_material():
        return "insufficient material"
    elif board.is_fivefold_repetition():
        return "fivefold repetition"
    else:
        return "Greg flipping the board"

def playChess(board):
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if board.turn:
            board.push(philPlayer(moves))
        else:
            board.push(gregPlayer(moves))
    print("The game ended with the score " + str(board.result()) + " on turn " + str(board.fullmove_number) + " due to " + gameOverReason(board))
    return(board)