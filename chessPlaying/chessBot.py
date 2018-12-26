# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:28:33 2018

@author: L L L L L
"""

import chess
import random
from moveLogic import MCTS

class Player():
    #player class used to play models against one another
    def __init__(self, board, model):
        self.tree = MCTS.MCST(board, 200, 0, 1, model)
        self.model = model
        self.board = board
        self.policies = []
        
    def nextMove(self):
        return self.tree.nextNode(1)
    
def deterministicPlayer(moves):
    #just picks the first move
    return moves[0]

def randomPlayer(moves):
    #random player
    return random.choice(moves)

def gameOverReason(board):
    #provides better debugging and information
    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.is_insufficient_material():
        return "insufficient material"
    elif board.is_fivefold_repetition():
        return "fivefold repetition"
    else:
        return "other reason"

def playChess(player1, player2, board):
    #logic for playing a game
    while not board.is_game_over():
        if board.fullmove_number > 80:
            #allows for ending a game prematurely to train faster
            player1.tree.t = .5
            player2.tree.t = .5
            break;
        moves = list(board.legal_moves)
        if board.turn:
            move, policy = player1.nextMove()
            board.push_uci(move)
            player1.policies += [policy]
            if player2.model == "random":
                continue
            else:
                #update the MCTS so that it still has the relevant part of the 
                #tree for the next move
                if player2.tree.head.edges is not None:
                    if move in player2.tree.head.edgeMoves:
                        player2.tree.head = player2.tree.head.edges[player2.tree.head.edgeMoves.index(move)].nxt
                        player2.tree.head.prev.nxt = None
                        player2.tree.head.prev = None
                    else:
                        player2.tree.head = MCTS.Node(None, move, None)
                else:
                    player2.tree.head = MCTS.Node(None, move, None)
        else:
            if player2.model == "random":
                move = randomPlayer(moves)
                board.push(move)
            else:
                move, policy = player2.nextMove()
                player2.policies += [policy]
                board.push_uci(move)
            if move in player1.tree.head.edgeMoves:
                player1.tree.head = player1.tree.head.edges[player1.tree.head.edgeMoves.index(move)].nxt
                player1.tree.head.prev.nxt = None
                player1.tree.head.prev = None
            else:
                player1.tree.head = MCTS.Node(None, move, None)
    print("The game ended with the score " + str(board.result()) + " on turn " + str(board.fullmove_number) + " due to " + gameOverReason(board))
    return(board)