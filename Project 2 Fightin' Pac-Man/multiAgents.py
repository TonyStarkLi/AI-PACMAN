# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # get all for positions
        foodPos = newFood.asList()
        # count how many foods
        foodCount = len(foodPos)
        # set default distance
        closestDistance = float("inf")

        # edge cases where there are no more food
        if foodCount == 0 :
          closestDistance = 0
        else:
          # for each food we cal the distance
          for i in xrange(foodCount):
            distance = manhattanDistance(foodPos[i],newPos)
            distance += foodCount*100
            # update distance: compare to each possible solution
            if distance < closestDistance:
              closestDistance = distance

        # print closestDistance
        score = -closestDistance

        # considering ghost pos
        for i in xrange(len(newGhostStates)):
          ghostPos = successorGameState.getGhostPosition(i+1)
          # if the ghost is next to pac man, then will never move to that direction
          if manhattanDistance(newPos,ghostPos)<=1 :
            score -= float("inf")

        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def getList(lst):
          return [x for x in lst if x != 'Stop']

        def miniMax(s, iterCount):
          # base cases
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return self.evaluationFunction(s)

          turn = iterCount%numAgent
          #Ghost min
          if iterCount%numAgent != 0:
            result = float("inf")
            for lst in getList(s.getLegalActions(turn)):
              result = min(result, miniMax(s.generateSuccessor(turn,lst), iterCount+1))
            return result

          # Pacman Max
          else: 
            result = float("-inf")
            for lst in getList(s.getLegalActions(turn)):
              result = max(result, miniMax(s.generateSuccessor(turn,lst), iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
          
        _ = miniMax(gameState, 0);
        return getList(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def getList(lst):
          return [x for x in lst if x != 'Stop']

        def alphaBeta(s, iterCount, alpha, beta):
          # base cases
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return self.evaluationFunction(s)

          turn = iterCount%numAgent
          #Ghost min
          if iterCount%numAgent != 0:
            result = 1e10
            for lst in getList(s.getLegalActions(turn)):
              result = min(result, alphaBeta(s.generateSuccessor(turn,lst), iterCount+1, alpha, beta))
              beta = min(beta, result)
              if beta < alpha:
                break
            return result

          # Pacman Max
          else: 
            result = -1e10
            for lst in getList(s.getLegalActions(turn)):
              result = max(result, alphaBeta(s.generateSuccessor(turn,lst), iterCount+1, alpha, beta))
              alpha = max(alpha, result)
              if iterCount == 0:
                ActionScore.append(result)
              if beta < alpha:
                break
            return result
          
        _ = alphaBeta(gameState, 0, -1e20, 1e20);
        return getList(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def getList(lst):
          return [x for x in lst if x != 'Stop']

        def expectimax(s, iterCount):
          # base cases
          if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
            return self.evaluationFunction(s)

          turn = iterCount%numAgent
          #Ghost min
          if iterCount%numAgent != 0:
            successorScore = []
            for lst in getList(s.getLegalActions(turn)):
              result = expectimax(s.generateSuccessor(turn,lst), iterCount+1)
              successorScore.append(result)
            averageScore = sum([float(x) / len(successorScore) for x in successorScore])
            return averageScore

          # Pacman Max
          else: 
            result = float("-inf")
            for lst in getList(s.getLegalActions(turn)):
              result = max(result, expectimax(s.generateSuccessor(turn,lst), iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
          
        _ = expectimax(gameState, 0);
        return getList(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()















