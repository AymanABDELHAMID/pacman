
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

# from .util import *
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]



"""
The following implementation is taken from the email sent by the professor 28/04/2020
"""


class Node:
    nodeCount = 0

    def __init__(self, state, action, parent, cost):
        """
        """
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost
        self.no = Node.nodeCount
        Node.nodeCount += 1

    def isEqual(self, state):
        # When is a node equal to another ?
        pass

    def isNotEqual(self, other):
        # When is a node not equal to another ?
        return not self.__eq__(other)

    def isLess(self, other):
        # When is a node less than another ?
        return self.cost < other.cost

    def showNode(self):
        # How to print a Node in a human readable way
        return '<' + str(self.state) + ',' + str(self.action) + ',' + str(self.cost) + '>'

    def numNodes(self):
        # displays number of nodes visited
        return '< Number of nodes visisted in this Maze,' + str(self.no) + '>'

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    Update 11/05 - I get an error in the nodes expanded if I leave the print.
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    ##########################
    """
    This algorithm is inspired from the lecture and the reference, algorithm pseudocode can be found in Chapter 4
    Figure 4.21 
    """
    "Stack() = A container with a last-in-first-out (LIFO) queuing policy."

    fringe = util.Stack()
    fringe.push((problem.getStartState(), [], []))

    while not fringe.isEmpty():
        state, action, parent = fringe.pop()  # Pop the most recently pushed item from the stack
        node = Node(state, action, parent, problem.getCostOfActions(action))

        if problem.isGoalState(node.state):
            "print(node.showNode())"
            print(node.numNodes())
            return node.action # path # try node.action

        for child, direction, steps in problem.getSuccessors(node.state):
            if not child in node.parent:
                fringe.push((child, node.action + [direction], node.parent + [node.state]))
    return []
    ##########################
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue()
    fringe.push((problem.getStartState(), [], []))
    expanded = [] # try to connect it with the Node class

    while not fringe.isEmpty():
        state, action, cost = fringe.pop()  # Pop the most recently pushed item from the stack
        node = Node(state, action, [], cost)
        "print(node.state)"

        if not node.state in expanded:
            expanded.append(node.state)

            if problem.isGoalState(node.state):
                "print(node.showNode())"
                print(node.numNodes())
                return node.action

            for child, direction, c in problem.getSuccessors(node.state):
                fringe.push((child, node.action + [direction], node.cost + [c]))

    return []
    ##########################
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    fringe = util.PriorityQueue() # priority queue will allow me to use the automatic sorting of the stack
    fringe.push((problem.getStartState(), [], 0), 0)
    expanded = []  # try to connect it with the Node class

    while not fringe.isEmpty():
        state, action, cost = fringe.pop()  # Pop the most recently pushed item from the stack
        node = Node(state, action, [], cost)

        if not node.state in expanded:
            expanded.append(node.state)

            if problem.isGoalState(node.state):
                "print(node.showNode())"
                print(node.numNodes())
                return node.action

            for child, direction, c in problem.getSuccessors(node.state):
                fringe.push((child, node.action + [direction], node.cost + c), node.cost + c)

    return []
    ##########################
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    fringe = util.PriorityQueue()
    fringe.push( (problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) )
    expanded = []  # try to connect it with the Node class

    while not fringe.isEmpty():
        state, action, cost = fringe.pop()  # Pop the most recently pushed item from the stack
        node = Node(state, action, [], cost)

        if not node.state in expanded:
            expanded.append(node.state)

            if problem.isGoalState(node.state):
                "print(node.showNode())"
                print(node.numNodes())
                return node.action

            for child, direction, c in problem.getSuccessors(node.state):
                temp = node.cost + c
                fringe.push((child, node.action + [direction], node.cost + c), temp + heuristic(child,problem))

    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

