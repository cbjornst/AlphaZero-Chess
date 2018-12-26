# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:15:30 2018

@author: L L L L L
"""
import chess
import time
from math import sqrt
import numpy as np
class Edge:
    #MCTS edge
    def __init__(self, a, P, prev, nxt):
        self.a = a
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = P
        self.prev = prev
        self.nxt = nxt
        
class Node:
    #MCTS Node
    def __init__(self, edges, mv, prev):
        self.edges = edges
        self.edgeMoves = []
        self.mv = mv
        self.prev = prev
        
class MCST:
    def __init__(self, s, trials, turn, t, model):
        #Trials is usually set to 75, t is set to 1
        self.s = s
        self.turn = turn
        self.trials = trials
        if s.move_stack != []:
            self.head = Node(None, s.peek(), None)
        else:
            self.head = Node(None, "", None)
        self.t = t
        self.model = model
        
    def getProb(self, legalMoves, probs, board, training, Q):
        #Get the probabilities from the neural net output, or alternatively
        #convert the calculated probabilites to a neural net input 
        prob = []
        for move in legalMoves:
            fr = move.from_square
            to = move.to_square
            y1 = to // 8
            y2 = fr // 8
            x1 = to % 8
            x2 = fr % 8
            if board.san(move)[0].lower() == "n":
                if y1 - y2 == 2:
                    if x1 - x2 == 1:
                        z = 56
                    else:
                        z = 57
                elif y1 - y2 == 1:
                    if x1 - x2 == 2:
                        z = 58
                    else:
                        z = 59
                elif y1 - y2 == -1:
                    if x1 - x2 == 2:
                        z = 60
                    else:
                        z = 61
                else:
                    if x1 - x2 == 1:
                        z = 62
                    else:
                        z = 63
            elif board.san(move)[-1].lower() in ["n", "b", "r"]:
                promo = board.san(move)[-1].lower()
                if promo == "n":
                    z = 0
                elif promo == "b":
                    z = 3
                else:
                    z = 6
                if x1 > x2:
                    z += 64
                elif x1 < x2:
                    z += 65
                else:
                    z += 66     
            else:
                dist = chess.square_distance(fr, to)
                if y1 > y2:
                    if x1 > x2:
                        z = 7
                    elif x1 < x2:
                        z = 49
                    else:
                        z = 0
                elif y1 < y2:
                    if x1 > x2:
                        z = 21
                    elif x1 < x2:
                        z = 35
                    else:
                        z = 28
                else:
                    if x1 > x2:
                        z = 14
                    else:
                        z = 42
                z += dist
            if training:
                probs[z][y2][x2] = Q[legalMoves.index(move)]
            else:
                p = probs[z][y2][x2]    
                prob += [p]
        if training:
            return probs
        else:
            softmax = np.exp(prob)
            softmax = softmax / np.sum(softmax)
            return softmax
    
    def nextNode(self,c):
        #find the next node
        start = time.clock()
        board = self.s.copy()
        for i in range(self.trials):
            currentNode = self.head
            depth = 0
            turn = 1
            while currentNode.edges is not None and currentNode.edges != [] and depth < 6:
                #search until an unexplored node is found or depth is reached
                dr = np.random.dirichlet([.03] * len(currentNode.edges))
                depth += 1
                turn *= -1
                utility = []
                UCT = 0
                #UCT = sum(edge.N for edge in currentNode.edges)
                for i in range(len(currentNode.edges)):
                    UCT += currentNode.edges[i].N
                for i in range(len(currentNode.edges)):
                    PUCT = sqrt(UCT) / (1 + currentNode.edges[i].N)
                    Q = currentNode.edges[i].Q
                    if currentNode is self.head:
                        U = c * ((currentNode.edges[i].P * .66) + (.33 * dr[i])) * PUCT
                    else:
                        U = c * currentNode.edges[i].P * PUCT
                    utility += [Q + U]
                nextEdge = currentNode.edges[utility.index(max(utility))]
                currentNode = nextEdge.nxt
                board.push_uci(currentNode.mv)
            nnInput = self.model.parseInput(board, 2)
            probs, v = self.model.runModel(nnInput)
            v = v[0][0]
            probs = probs[0].reshape(73, 8, 8)
            legalMoves = list(board.legal_moves)
            p = self.getProb(legalMoves, probs, board, False, None)
            edgeList = []
            #add new nodes
            for i in range(len(legalMoves)):
                edgeList += [Edge(legalMoves[i], p[i], currentNode, None)]
                nextMove = str(legalMoves[i])
                edgeList[i].nxt = Node(None, nextMove, edgeList[i]) 
                currentNode.edgeMoves = [str(move) for move in legalMoves]
            currentNode.edges = edgeList
            self.backpropogate(currentNode, v, turn)
            while depth != 0:
                turn *= -1
                depth -= 1
                board.pop()
        #print(time.clock() - start)
        move = self.pickMove()
        return move

    def backpropogate(self, node, v, turn):
        #update edges
        while node.prev is not None:
            turn = turn * -1
            edge = node.prev
            edge.N += 1
            edge.W = edge.W + (v * turn)
            edge.Q = edge.W / edge.N
            node = edge.prev

    def pickMove(self):
        #pick best move using visit count
        policy = []
        edges = self.head.edges
        denom = 0
        for i in range(len(edges)):
            denom += edges[i].N ** (1 / self.t)
        for i in range(len(edges)):
            policy += [(edges[i].N ** (1 / self.t)) / denom] 
        nextHead = edges[policy.index(max(policy))].nxt
        move = nextHead.mv
        self.head = nextHead
        self.head.prev.nxt = None
        self.head.prev = None
        return move, policy