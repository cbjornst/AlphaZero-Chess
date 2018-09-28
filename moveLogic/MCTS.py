# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:15:30 2018

@author: L L L L L
"""
import chess

class Edge:
    def __init__(self, a, P, prev):
        self.a = a
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = P
        self.prev = prev
        
class Node:
    def __init__(self, edges, s, prev):
        self.edges = edges
        self.s = s
        self.prev = prev
        
class MCST:
    def __init__(self, s, depth, turn):
        self.s = s
        self.turn = turn
        self.depth = depth
    
    def nextNode(self, s, turn):
        head = Node(None, s, None)
        for i in range(self.depth):
            while head.edges is not None:
                utility = []
                for i in range(len(head.edges)):
                    Q = head.edges[i].Q
                    U = c * head.edges[i].P * UCT
                    utiliy += [Q + U]
                    
                    
                    
                    s.push(head.edges[i])
                    self.nextNode(s, self.depth - 1, not turn)
                    s.pop()
                    
    def backpropogate(self, node, v):
        while node.prev is not None:
            edge = node.prev
            edge.N += 1
            edge.W = edge.W + v
            edge.Q = (edge.W / edge.N)
            node = edge.prev
