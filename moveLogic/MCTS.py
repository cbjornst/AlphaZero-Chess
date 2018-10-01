# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:15:30 2018

@author: L L L L L
"""
import chess
from moveLogic import neuralNet

class Edge:
    def __init__(self, a, P, prev, nxt):
        self.a = a
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = P
        self.prev = prev
        self.nxt = nxt
        
class Node:
    def __init__(self, edges, s, prev):
        self.edges = edges
        self.s = s
        self.prev = prev
        
class MCST:
    def __init__(self, s, trials, turn, t):
        self.s = s
        self.turn = turn
        self.trials = trials
        self.head = Node(None, s, None)
        self.t = t
    
    def nextNode(self, s, turn, c):
        for i in range(self.trials):
            currentNode = self.head
            while currentNode.edges is not None:
                utility = []
                UCT = 0
                for i in range(len(currentNode.edges)):
                    UCT += currentNode.edges[i].N
                for i in range(len(currentNode.edges)):
                    PUCT = UCT / (1 + currentNode.edges[i].N)
                    Q = currentNode.edges[i].Q
                    U = c * currentNode.edges[i].P * PUCT
                    utility += [Q + U]
                s.push(self.head.edges[i])
                nextEdge = max(utility)
                currentNode = nextEdge.nxt
            nnInput = neuralNet.parseInput(currentNode.s, 8)
            probs, v = neuralNet.runModel(nnInput)
            newEdges = list(s.legal_moves)
            edgeList = []
            for i in range(len(newEdges)):
                edgeList[i] = Edge(newEdges[i], probs[i], currentNode, None)
                edgeList[i].nxt = Node(None, s, edgeList[i]) 
            currentNode.edges = edgeList
            self.backpropogate(currentNode, v)
            print(probs, v)
    
    def backpropogate(self, node, v):
        while node.prev is not None:
            edge = node.prev
            edge.N += 1
            edge.W = edge.W + v
            edge.Q = (edge.W / edge.N)
            node = edge.prev

    def pickMove(self):
        policy = []
        edges = self.head.edges
        denom = 0
        for i in range(len(edges)):
            denom += edges[i].N ^ (1 / self.t)
        for i in range(len(edges)):
            policy += [(edges[i].N ^ (1 / self.t)) / denom] 
        move = max(policy)
        self.head = edges[max(policy)].next
        self.prev = None
        return move