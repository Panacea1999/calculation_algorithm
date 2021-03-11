#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: A_star.py
@Abstract: A-Star search
@Time: 2021/03/11 09:34:27
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''
from typing import Generic, TypeVar, Callable
from typing_extensions import Protocol
from heapq import heappush, heappop

T = TypeVar('T')

class PriorityQueue(Generic[T]):
    def __init__(self):
        self._container = []
    
    @property
    def empty(self):
        return not self._container
    
    def push(self,item):
        heappush(self._container, item)

    def pop(self):
        return heappop(self._container)

    def __repr__(self):
        return repr(self._container)

def manhattan_distance(goal):
    def distance(location):
        xdist = abs(location.column - goal.column)
        ydist = abs(location.row - goal.row)
        return (xdist + ydist)
    return distance

class Node(Generic[T]):
    def __init__(self, state, parent, cost=0.0, heuristic=0.0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic= heuristic
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def astar(initial, goal_test, successors, heuristic):
    frontier = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    explored = {initial: 0.0}
    while not frontier.empty:
        current_node = frontier.pop()
        current_state = current_node.state
        # goal test
        if goal_test(current_state):
            print(current_node.state)
            return current_node
        # check for next location
        for child in successors(current_state):
            new_cost = current_node.cost + 1
            if child not in explored or explored[child] > new_cost:
                explored[child] = new_cost
                frontier.push(Node(child, current_node, new_cost, heuristic(child)))
    return None

def node_to_path(node):
    path = [node.state]
    while node.parent is not None:
        node = node.parent
        path.append(node.state)
    path.reverse()
    return path

