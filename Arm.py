import numpy as np
from sklearn import preprocessing
from discreteMarkovChain import markovChain
import pykov
import networkx as nx

class Arm:

    def __init__(self, num_states, rand=True, arm_count=0):
        self.num_states = num_states

        self.cumulative_reward = 0.0
        self.pull_count = 0

        self.rand = rand
        self.arm_count = arm_count

        if self.rand:
            # create ergodic transition matrix
            self.tran_matrix = np.zeros((num_states, num_states))
            ergodic = False
            while ergodic == False:
                P = np.random.rand(num_states, num_states)
                P = preprocessing.normalize(P, norm='l1', axis=1, copy=True, return_norm=False)
                dic = {}
                for i in range(num_states):
                    for j in range(num_states):
                        dic[(i,j)] = P[i,j]
                T = pykov.Chain(dic)
                G = nx.DiGraph(list(T.keys()))
                if nx.is_strongly_connected(G) and nx.is_aperiodic(G):
                    ergodic = True
                    self.tran_matrix = P
        else:
            if self.arm_count == 0:
                self.tran_matrix = np.array([[0., 1.], [.01, .99]])
            if self.arm_count == 1:
                self.tran_matrix = np.array([[.99, .01], [1., 0.]])

        # find the steady state distribution
        mc = markovChain(self.tran_matrix)
        mc.computePi('linear') # We can also use 'power', 'krylov' or 'eigen'
        self.steady_state = np.asmatrix(mc.pi)

        # find ergodic constants
        P_tilde = np.zeros((self.num_states, self.num_states)) # P_tilde as per Formula 2.1 in Page 65
        for x in range(self.num_states):
            for y in range(self.num_states):
                P_tilde[x,y] = self.steady_state[0, y] * self.tran_matrix[y,x] / self.steady_state[0, x]

        MP = np.dot(self.tran_matrix, P_tilde)
        eigenval, _ = np.linalg.eig(MP)
        self.eta = np.sort(eigenval)[-2] # Second largest eigenvalue
        self.eta = np.sqrt(self.eta) # sqrt for TV norm

        # Now we need to compute C using chi_0 square in Thm 2.7
        # Here we don't have information about current state,
        # given a steady state distribution, chi_0 is maximized when current state is of the form (1,0,0,0,0,0)
        curr_state = np.zeros((1, self.num_states))
        index = np.argmin(self.steady_state)
        curr_state[0, index] = 1.0

        self.C = 0.25 * sum([(self.steady_state[0, i] - curr_state[0, i]) ** 2 / self.steady_state[0, i] for i in range(self.num_states)])
        self.C = np.sqrt(self.C) # sqrt for TV norm

        if self.rand:
            self.reward_type = np.random.randint(0, 3)
            self.set_reward_parameters()
        else:
            if self.arm_count == 0:
                self.rewards = np.array([0, 1])
            elif self.arm_count == 1:
                self.rewards = np.array([.5, .5])


    def set_reward_parameters(self):
        if self.reward_type == 0:
            self.reward_parameters = np.zeros((self.num_states, 3))
            for i in xrange(self.num_states):
                gamma = np.random.uniform(0, 5)  # This needs to be positive.
                beta = np.random.uniform(0, 5)  # This needs to be positive.
                mean_beta = gamma/(gamma + beta)
                self.reward_parameters[i] = [gamma, beta, mean_beta]
        elif self.reward_type == 1:
            self.reward_parameters = np.zeros((self.num_states, 3))
            for i in xrange(self.num_states):
                a = np.random.uniform(0, .5)  # This needs to be in [0, .5].
                b = np.random.uniform(.5, 1)  # This needs to be in [.5, 1].
                mean_uniform = .5*(a + b)
                self.reward_parameters[i] = [a, b, mean_uniform]
        elif self.reward_type == 2:
            self.reward_parameters = np.zeros((self.num_states, 2))
            for i in xrange(self.num_states):
                p = np.random.uniform(0, .5)  # This needs to be in [0, 1].
                mean_bernoulli = p
                self.reward_parameters[i] = [p, mean_bernoulli]


    def get_rewards(self, s):
        if not self.rand:
            return self.rewards[s]
        if self.reward_type == 0:
            return np.random.beta(self.reward_parameters[s, 0], self.reward_parameters[s, 1])
        elif self.reward_type == 1:
            return np.random.uniform(self.reward_parameters[s, 0], self.reward_parameters[s, 1])
        elif self.reward_type == 2:
            return np.random.binomial(1, self.reward_parameters[s, 0])


    def initialize_arm(self):
        self.cumulative_reward = 0.0
        self.pull_count = 0

    def get_tran_matrix(self):
        return self.tran_matrix

    def get_steady_state(self):
        return self.steady_state

    def get_expect_reward(self):
        if self.rand:
            return np.asscalar(np.dot(self.steady_state, self.reward_parameters[:, -1]))
        else:
            return np.asscalar(np.dot(self.steady_state, self.rewards))

    def get_ave_reward(self):
        if self.pull_count == 0:
            return 0.0
        return self.cumulative_reward / self.pull_count

    def get_pull_count(self):
        return self.pull_count

    def evolve(self, state):
        new_state = np.dot(state, self.tran_matrix)
        new_state = preprocessing.normalize(new_state, norm='l1', axis=1, copy=True, return_norm=False)
        return new_state

    def pull(self, state, iteration=1):
        cum_reward = 0.0
        for _ in range(iteration):
            # sample according to state distribution
            state_index = np.random.choice(np.arange(self.num_states), p=state.tolist()[0])
            reward = self.get_rewards(state_index)
            # reward = 0.0
            # if np.random.rand() < self.rewards[state_index][0]:
            #     reward = 1.0
            # else:
            #     reward = 0.0
            cum_reward += reward
            state = self.evolve(state)
        ave_reward = cum_reward / iteration
        self.cumulative_reward += ave_reward
        self.pull_count += 1
        return ave_reward, state

    def get_ergodic_constants(self, curr_state=np.zeros(())): 
        '''
            Returns Constants C, eta in that order
        '''
        if np.sum(curr_state) == 0:
            return self.C, self.eta
        # If we have information about current state
        else:
            C = 0.25 * sum([(self.steady_state[0, i] - curr_state[0, i]) ** 2 / self.steady_state[0, i] for i in range(self.num_states)])
            C = np.sqrt(C) # sqrt for TV norm

        return C, self.eta




