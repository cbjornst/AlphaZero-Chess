# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:28:33 2018

@author: L L L L L
"""

import chess
import random
from moveLogic import MCTS

class Player():
    def __init__(self, board, model):
        self.tree = MCTS.MCST(board, 3, 0, 1, model, 5)
        self.model = model
        self.board = board
    def nextMove(self):
        return self.tree.nextNode(self.board.turn, 1)
    
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

def playChess(player1, player2, board):
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if board.turn:
            board.push(player1.nextMove())
            print(board)
        else:
            board.push(gregPlayer(moves))
    print("The game ended with the score " + str(board.result()) + " on turn " + str(board.fullmove_number) + " due to " + gameOverReason(board))
    return(board)