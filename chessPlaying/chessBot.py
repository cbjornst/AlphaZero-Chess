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
        self.tree = MCTS.MCST(board, 50, 0, 1, model)
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

def playChess(player1, board):
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if board.turn:
            board.push_uci(player1.nextMove())
        else:
            move = gregPlayer(moves)
            if move in player1.tree.head.edges:
                player1.tree.head = player1.tree.head.edges[player1.tree.head.edges.index(move)].nxt
                player1.tree.head.prev.nxt = None
                player1.tree.head.prev = None
            else:
                player1.tree.head = MCTS.Node(None, move, None)
            board.push(move)
    print("The game ended with the score " + str(board.result()) + " on turn " + str(board.fullmove_number) + " due to " + gameOverReason(board))
    return(board)