#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@FileName: Maze.py
@Abstract: Maze class
@Time: 2021/03/11 08:05:33
@Requirements: 
@Author: WangZy ntu.wangzy@gmail.com
@Version: -
'''

import random
from typing import NamedTuple
from enum import Enum
from math import sqrt
from A_star import manhattan_distance, astar, node_to_path, Node

class Cell(str, Enum):
    EMPTY = ' '
    BLOCKED = 'X'
    START = 'S'
    GOAL = 'G'
    PATH = '@'

class MazeLocation(NamedTuple):
    row: int
    column: int

class Maze:
    def __init__(self, rows=10, columns=10, sparseness=0.2, start=MazeLocation(0, 0), goal=MazeLocation(9, 9)):
        # initialize instance variables
        self._rows = rows
        self._columns = columns
        self.start = start
        self.goal = goal
        self._grid = [[Cell.EMPTY for c in range(columns)] for r in range(rows)]
        # fill blocked cells
        self._randomly_fill(rows, columns, sparseness)
        # fill the start and goal
        self._grid[start.row][start.column] = Cell.START
        self._grid[goal.row][goal.column] = Cell.GOAL
    
    def _randomly_fill(self, rows, columns, sparseness):
        for row in range(rows):
            for column in range(columns):
                if random.uniform(0, 1.0) < sparseness:
                    self._grid[row][column] = Cell.BLOCKED
    
    #print
    def __str__(self):
        output = ''
        for row in self._grid:
            output += ''.join([c.value for c in row]) + '\n'
        return output

    def goal_test(self, location):
        return location == self.goal

    def successors(self, location):
        locations = []
        if location.row + 1 < self._rows and self._grid[location.row+1][location.column] != Cell.BLOCKED:
            locations.append(MazeLocation(location.row+1, location.column))
        if location.row - 1 >= 0 and self._grid[location.row-1][location.column] != Cell.BLOCKED:
            locations.append(MazeLocation(location.row-1, location.column))
        if location.column - 1 >= 0 and self._grid[location.row][location.column-1] != Cell.BLOCKED:
            locations.append(MazeLocation(location.row, location.column-1))
        if location.column + 1 < self._columns and self._grid[location.row][location.column+1] != Cell.BLOCKED:
            locations.append(MazeLocation(location.row, location.column+1))
        return locations
    
    def mark(self, path):
        for maze_location in path:
            self._grid[maze_location.row][maze_location.column] = Cell.PATH
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL
    
    def clear(self, path):
        for maze_location in path:
            self._grid[maze_location.row][maze_location.column] = Cell.EMPTY
        self._grid[self.start.row][self.start.column] = Cell.START
        self._grid[self.goal.row][self.goal.column] = Cell.GOAL

#test
m = Maze()
distance = manhattan_distance(m.goal)
solution = astar(m.start, m.goal_test, m.successors,distance)
print(solution.state)
if solution is None:
    print('No Solution')
else:
    path = node_to_path(solution)
    m.mark(path)
    print(m)
