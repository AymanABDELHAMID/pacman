# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(0, self.iterations):
            # import copy
            values_temp = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    values_temp[state] = 0
                    continue

                maxActionValue = float('-inf')
                possibleActions = self.mdp.getPossibleActions(state)

                for action in possibleActions:
                    actionSumSPrime = self.getQValue(state, action)

                    "Find the maximum action"
                    if maxActionValue < actionSumSPrime:
                        maxActionValue = actionSumSPrime

                v_kPlus1 = maxActionValue
                values_temp[state] = v_kPlus1
            self.values = values_temp

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          Q*(s,a) = sum[s'] T(s,a,s')[R(s,a,s')+a.V_{k}(s)]
        """
        actionSumSPrime = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            TransitionProb = transition[1]
            statePrime = transition[0]
            gamma = self.discount  # the discount factor
            reward = self.mdp.getReward(state, action, statePrime)
            actionSumSPrime += TransitionProb * (reward + (gamma * self.values[statePrime]))

        return actionSumSPrime

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          V_{k+1}(s) = max[a] Q*(s,a)
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            actions = self.mdp.getPossibleActions(state)
            max_value = self.getQValue(state, actions[0])
            max_action = actions[0]

            for action in actions:
                value = self.getQValue(state, action)
                if max_value <= value:
                    max_value = value
                    max_action = action

            return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        iterations = self.iterations
        states = (self.mdp).getStates()
        for i in range(iterations):
            index = i % len(states)
            if not (self.mdp).isTerminal(states[index]):
                bestAction = self.computeActionFromValues(states[index])
                self.values[states[index]] = self.computeQValueFromValues(states[index], bestAction)
        return


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessor = dict.fromkeys((self.mdp).getStates(), set())
        qDict = dict()
        for state in (self.mdp).getStates():
            # Compute predecessors of all states
            actions = (self.mdp).getPossibleActions(state)
            for a in actions:
                successor = (self.mdp).getTransitionStatesAndProbs(state, a)[0][0]
                predecessor[successor].add(state)
        pq = util.PriorityQueue()
        for state in (self.mdp).getStates():
            if not (self.mdp).isTerminal(state):
                actions = (self.mdp).getPossibleActions(state)
                qvalues = [self.computeQValueFromValues(state, a) for a in actions]
                diff = abs(self.getValue(state) - max(qvalues))
                qDict[state] = max(qvalues)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                # If the priority queue is empty, then terminate
                return
            s = pq.pop() # Pop a state s off the priority queue
            if not (self.mdp).isTerminal(s):
                # Update s's value (if it is not a terminal state) in self.values
                self.values[s] = qDict[s]
            for p in predecessor[s]:
                actions = (self.mdp).getPossibleActions(p)
                qvalues = [self.computeQValueFromValues(p, a) for a in actions]
                diff = abs(self.values[p] - max(qvalues))
                if diff > self.theta:
                    qDict[p] = max(qvalues)
                    pq.update(p, -diff)
