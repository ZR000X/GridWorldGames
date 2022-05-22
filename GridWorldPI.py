import random
from typing import overload
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, xdim, ydim, state=None):
        self.xdim = xdim
        self.ydim = ydim
        self.size = xdim * ydim
        if state is None:
            self.go_random_state()
        else:
            self.state = state
            
    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = [state[0] % self.xdim, state[1] % self.ydim]
    
    def go_random_state(self, max_tries=None):
        if max_tries is None:
            max_tries = self.size
        self.state = [random.choice(range(self.xdim)),random.choice(range(self.ydim))]

    ### Movements
    def up(self, state):
        if type(state) is int:
            state = self.number_state(state)
        if state[1] > 0:
            return [state[0], state[1] - 1]
        return state
        
    def down(self, state):
        if type(state) is int:
            state = self.number_state(state)
        if state[1] < self.ydim - 1:
            return [state[0], state[1] + 1]
        return state
    
    def left(self, state):
        if type(state) is int:
            state = self.number_state(state)
        if state[0] > 0:
            return [state[0] - 1, state[1]]
        return state
    
    def right(self, state):
        if type(state) is int:
            state = self.number_state(state)
        if state[0] < self.xdim - 1:
            return [state[0] + 1, state[1]]
        return state
    
    def copy(self, random_state = False):
        return GridWorld(self.xdim, self.ydim, self.state if not random_state else None)
    
    def state_number(self, state):
        return state[1] * self.xdim + state[0]
    
    def number_state(self, n):
        x = n % self.xdim
        y = (n - x) / self.xdim
        return [int(x), int(y)]

class GetOutGame(GridWorld):
    """
    Implements the example found on pg9,10,... of https://www.davidsilver.uk/wp-content/uploads/2020/03/DP.pdf    
    """
    def __init__(self, xdim, ydim, terminals, state=None):
        init_state = state is None
        super().__init__(xdim=xdim, ydim=ydim, state=[0,0] if init_state else state)
        self.terminals = list()
        print(type(terminals[0]))
        if type(terminals[0]) is list or type(terminals[0]) is np.ndarray:
            for t in terminals:
                self.terminals.append([t[0]%xdim, t[1]%ydim])
        else:
            for t in terminals:
                self.terminals.append(self.number_state(t))
        self.actions = ['n', 's', 'e', 'w']
        self.next_state = {
            'n': self.up,
            's': self.down,
            'e': self.right,
            'w': self.left                    
        }
        if init_state:
            self.go_random_state()

    def go_random_state(self, max_tries=None):
        if max_tries is None:
            max_tries = self.size
        self.state = [random.choice(range(self.xdim)),random.choice(range(self.ydim))]
        i = 0
        while self.state in self.terminals and i < max_tries:
            self.set_state([random.choice(range(self.xdim)),random.choice(range(self.ydim))])
            i += 1
        
    def get_state_values(self, N=100, gamma=1):
        """
        We use iterative policy evaluation (IPE) to compute the value of each non-terminal state
        Each iteration, we use the Bellman equation to update the values over state-space
        We use greeedy policy in each iteration
        """
        # Data storage
        all_states = np.zeros(self.size)
        state_values = np.zeros(self.size)
        new_state_values = np.zeros(self.size)
        for _ in range(N):  # iteration is done N times            
            for state_index in range(len(all_states)): # we update each statevalue
                state = self.number_state(state_index)
                # Check if state is terminal
                if state in self.terminals:
                    continue
                next_states = dict()
                # Check next states per action
                for a in self.actions:
                    next_states[a] = self.next_state[a](state)                
                # Compute expected value per action
                expected_action_value = dict()
                for action in next_states.keys():
                    expected_action_value[action] = state_values[self.state_number(next_states[action])]                
                # Determine what the policy would be: i.e., distribution over actions. We follow greedy policy.
                max_actions = []
                for action, value in expected_action_value.items():
                    if value == max(expected_action_value.values()):
                        max_actions.append(action)                
                policy_coefficient_per_action = dict()
                for action in next_states:
                    policy_coefficient_per_action[action] = 1/len(max_actions) if action in max_actions else 0            
                # Compute new state value to be assigned to be value function
                new_state_value = 0
                for action, policy_coeff in policy_coefficient_per_action.items():
                    ###### BELLMAN EQUATION
                    new_state_value += policy_coeff*(-1+gamma*state_values[self.state_number(next_states[action])])
                    ######
                new_state_values[state_index] = new_state_value
            state_values = new_state_values
            new_state_values = np.zeros(self.size)
        return state_values

    def get_grid_values(self, state_values=None):
        if state_values is None:
            state_values = self.get_state_values()
        a = np.array([[0]*self.xdim]*self.ydim)
        for y in range(self.ydim):
            for x in range(self.xdim):               
                a[y, x] = state_values[self.state_number([x,y])]
        return a

    def print_state_values(self, state_values=None):
        if state_values is None:
            state_values = self.get_state_values()
        plt.imshow(self.get_grid_values(state_values), cmap='viridis')
        plt.colorbar()
        plt.show()