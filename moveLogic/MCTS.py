# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:15:30 2018

@author: L L L L L
"""
import chess
import time
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
    def __init__(self, edges, mv, prev):
        self.edges = edges
        self.mv = mv
        self.prev = prev
        
class MCST:
    def __init__(self, s, trials, turn, t, model):
        self.s = s
        self.turn = turn
        self.trials = trials
        if s.move_stack != []:
            self.head = Node(None, s.peek(), None)
        else:
            self.head = Node(None, "", None)
        self.t = t
        self.model = model
        
    def getProb(self, move, probs, board):
        #TODO: Find a better way to calculate p than a series of if statements
        #I'm almost positive there's a better way to do this
        #but this works and the neural net is much more important right now
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
                    z = 58
            elif y1 - y2 == 1:
                if x1 - x2 == 2:
                    z = 59
                else:
                    z = 60
            elif y1 - y2 == -1:
                if x1 - x2 == 2:
                    z = 61
                else:
                    z = 62
            else:
                if x1 - x2 == 1:
                    z = 63
                else:
                    z = 64
            p = probs[z][y1][x1]
        elif board.san(move)[-1].lower() in ["q", "n", "b", "r"]:
            p = 0
        else:
            dist = chess.square_distance(fr, to)
            if y1 > y2:
                if x1 > x2:
                    z = 7 + dist
                elif x1 < x2:
                    z = 49 + dist
                else:
                    z = 0 + dist
            elif y1 < y2:
                if x1 > x2:
                    z = 21 + dist
                elif x1 < x2:
                    z = 35 + dist
                else:
                    z = 28 + dist
            else:
                if x1 > x2:
                    z = 14 + dist
                else:
                    z = 42 + dist
            p = probs[z][y1][x1]
        return p
    
    def nextNode(self, turn, c):
        start = time.clock()
        board = self.s.copy()
        for i in range(self.trials):
            currentNode = self.head
            depth = 0
            while currentNode.edges is not None and depth < 6:
                depth += 1
                utility = []
                UCT = 0
                #UCT = sum(edge.N for edge in currentNode.edges)
                for i in range(len(currentNode.edges)):
                    UCT += currentNode.edges[i].N
                for i in range(len(currentNode.edges)):
                    PUCT = UCT / (1 + currentNode.edges[i].N)
                    Q = currentNode.edges[i].Q
                    U = c * currentNode.edges[i].P * PUCT
                    utility += [Q + U]
                if utility == []:
                    print(currentNode.edges)
                nextEdge = currentNode.edges[utility.index(max(utility))]
                currentNode = nextEdge.nxt
                board.push_uci(currentNode.mv)
            nnInput = self.model.parseInput(board, 8)
            probs, v = self.model.runModel(nnInput)
            v = v[0][0]
            probs = probs[0].reshape(73, 8, 8)
            newEdges = list(board.legal_moves)
            edgeList = []
            if len(newEdges) > 0:
                for i in range(len(newEdges)):
                    edgeList += [Edge(newEdges[i], self.getProb(newEdges[i], probs, board), currentNode, None)]
                    nextMove = str(newEdges[i])
                    edgeList[i].nxt = Node(None, nextMove, edgeList[i]) 
            currentNode.edges = edgeList
            self.backpropogate(currentNode, v)
            while depth != 0:
                depth -= 1
                board.pop()
        print(time.clock() - start)
        move = self.pickMove()
        return move
    
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
            denom += edges[i].N ** (1 / self.t)
        for i in range(len(edges)):
            policy += [(edges[i].N ** (1 / self.t)) / denom] 
        nextHead = edges[policy.index(max(policy))].nxt
        move = nextHead.mv
        self.head = nextHead
        self.head.prev.nxt = None
        self.head.prev = None
        return move