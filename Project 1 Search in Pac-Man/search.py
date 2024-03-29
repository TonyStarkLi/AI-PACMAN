# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

# ______________________________________________________________________________


class Problem(object):
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.getSuccessors(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = action[0]
        return Node(next_state, self, action[1], self.path_cost + action[2])
        # next_state = problem.result(self.state, action)
        # next_node = Node(next_state, self, action,
        #                  problem.path_cost(self.path_cost, self.state,
        #                                    action, next_state))
        # return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


# ______________________________________________________________________________


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def generalGraphSearch(problem, fringe):
    """
    A general algorithm for searching
    """
    node = Node(problem.getStartState())
    fringe.push(node)
    visited = set()
    while not fringe.isEmpty():

        node = fringe.pop()
        curr_state = node.state

        if problem.isGoalState(curr_state):
            return node.solution()

        if curr_state not in visited:
            visited.add(curr_state)

            for sub_node in node.expand(problem):
                if sub_node.state not in visited:
                    fringe.push(sub_node)

    return False

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    return generalGraphSearch(problem, stack)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    return generalGraphSearch(problem, queue)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    
    # init
    fringe = util.PriorityQueue()
    visited = set()

    node = Node(state=problem.getStartState())
    fringe.push(node, node.path_cost)

    while not fringe.isEmpty():

        # get the min-cost node
        node = fringe.pop()
        curr_state = node.state

        if (curr_state in visited):
            continue

        visited.add(curr_state)

        if problem.isGoalState(curr_state):
            return node.solution()

        for sub_node in node.expand(problem):
            if sub_node.state not in visited:
                fringe.push(sub_node, sub_node.path_cost)

    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    # init
    fringe = util.PriorityQueue()
    visited = set()

    node = Node(state=problem.getStartState())
    fringe.push(node, node.path_cost)

    while not fringe.isEmpty():

        # get the min-cost node
        node = fringe.pop()
        curr_state = node.state

        if (curr_state in visited):
            continue

        visited.add(curr_state)

        if problem.isGoalState(curr_state):
            return node.solution()

        for sub_node in node.expand(problem):
            if sub_node.state not in visited:
                fringe.push(sub_node, sub_node.path_cost + heuristic(sub_node.state, problem))

    return None
    # fringe = util.PriorityQueue()
    # visited = set()

    # node = Node(state=problem.getStartState(), path_cost=heuristic(problem.getStartState(), problem))
    # fringe.push(node, node.path_cost)

    # while not fringe.isEmpty():

    #     node = fringe.pop()
    #     curr_state = node.state

    #     if (curr_state in visited):
    #         continue

    #     visited.add(curr_state)

    #     if problem.isGoalState(curr_state):
    #         return node.solution()

    #     for successor in problem.getSuccessors(curr_state):

    #         next_state = successor[0]
    #         next_action = successor[1]
    #         next_cost = successor[2]
    #         cost = next_cost + node.path_cost + heuristic(next_state, problem)

    #         next_node = Node(state=next_state, parent=node, action=next_action, path_cost=cost)

    #         if next_state not in visited:
    #             fringe.push(next_node, cost)

    # return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
