from Arm import *
import time
import numpy as np

class Bandit:

    def __init__(self, num_states, num_arms, horizon=1000, tau0=20, a=1.0, 
                 print_progress=False, stop_early=False, rand=True):
        self.num_states = num_states
        self.num_arms = num_arms
        self.horizon = horizon
        self.a = a # how much to increase each epoch
        self.tau0 = int(tau0) # number of iterations to run in each epoch
        self.print_progress = print_progress
        self.stopping_threshold = 5000
        self.stop_early = stop_early

        # create arms and compute expected rewards
        if rand:
            self.arms = [Arm(num_states) for _ in range(num_arms)] # create arms
        else:
            self.arms = [Arm(num_states, rand, _) for _ in range(num_arms)] # create arms
        self.optimal_expect_reward = max([arm.get_expect_reward() for arm in self.arms])
        self.best_arm_index = np.argmax([arm.get_expect_reward() for arm in self.arms])
        self.best_arm = self.arms[self.best_arm_index]
        self.expect_reward_sort = np.sort([arm.get_expect_reward() for arm in self.arms])
        self.min_delta = self.expect_reward_sort[-1] - self.expect_reward_sort[-2]

        # initiate state
        if rand:
            self.state = np.random.rand(1, num_states)
            self.state = preprocessing.normalize(self.state, norm='l1', axis=1, copy=True, return_norm=False)
        else:
            self.state = np.array([[0, 1]])

    def get_curr_state(self):
        return self.state

    def get_expected_rewards(self):
        return [arm.get_expect_reward() for arm in self.arms]

    def get_best_arm_index(self):
        return self.best_arm_index

    def get_optimal_expect_reward(self):
        return self.optimal_expect_reward

    def get_min_delta(self):
        return self.min_delta

    def ucb_choose_arm(self, n):
        ucbs = []
        for arm in self.arms:
            C, eta = arm.get_ergodic_constants()
            k = arm.get_pull_count()
            L = 0.5 * C / (1 - eta) * (1 / (self.tau0 + self.a) + 1 / self.a * np.log(1 + k * self.a / self.tau0))
            # ucb = arm.get_ave_reward() + L / self.tau0 + np.sqrt(8 * np.log(n) / arm.get_pull_count())
            ucb = arm.get_ave_reward() + L / k + np.sqrt(6 * np.log(n) / k)
            ucbs.append(ucb)
        return np.argmax(ucbs)

    def ucb_regret_bound(self, n):
        regret_bound = 0.0
        for arm in self.arms:
            if arm == self.best_arm:
                C, eta = arm.get_ergodic_constants()
                regret_bound += C / (1 - eta) / self.a * (1 + np.log(self.a * (n - 1) / self.tau0 + 1))
            else:
                C, eta = arm.get_ergodic_constants()
                delta = self.optimal_expect_reward - arm.get_expect_reward()
                gamma = 0.5 * C / (1 - eta) * (self.tau0 / (self.a + self.tau0) + np.sqrt(self.tau0 / self.a))
                t1 = 4 / (delta**2) * (gamma / self.tau0 + np.sqrt(6 * np.log(n)))**2 + 2*(np.log(n))
                t2 = delta + C / self.tau0 / (1 - eta)
                regret_bound += t1 * t2
        return regret_bound

    def UCB(self):
        for arm in self.arms:
            arm.initialize_arm()

        cumulative_reward = 0.0
        cumulative_regret = 0.0
        # state_history = [self.state]
        ave_rewards_history = []
        arm_history = []
        cumu_regret_history = []
        regret_bound_history = []

        for n in range(self.num_arms):
            # iteration = int((self.a ** n) * self.tau0)
            iteration = int(self.tau0 + self.a * n)
            reward, self.state = self.arms[n].pull(self.state, iteration)
            # state_history.append(self.state)
            cumulative_reward += reward
            cumulative_regret += self.optimal_expect_reward - reward
            ave_rewards_history.append(cumulative_reward / (n + 1))
            arm_history.append(n)
            cumu_regret_history.append(cumulative_regret)
            regret_bound_history.append(self.ucb_regret_bound(n + 1))

        for n in range(self.num_arms, self.horizon):
            if self.print_progress:
                if n % 100 == 0:
                    print "iteration n =", n, "reward =", ave_rewards_history[-1]
            index = self.ucb_choose_arm(n + 1)
            # iteration = int((self.a ** n) * self.tau0)
            iteration = int(self.tau0 + self.a * n)
            if self.stop_early:
                if iteration > self.stopping_threshold:
                    iteration = self.stopping_threshold
            reward, self.state = self.arms[index].pull(self.state, iteration)
            # state_history.append(self.state)
            cumulative_reward += reward
            cumulative_regret += self.optimal_expect_reward - reward
            ave_rewards_history.append(cumulative_reward / (n + 1))
            arm_history.append(index)
            cumu_regret_history.append(cumulative_regret)
            regret_bound_history.append(self.ucb_regret_bound(n + 1))

        self.ave_rewards_history = ave_rewards_history
        self.arm_history = arm_history
        self.cumu_regret_history = cumu_regret_history
        self.regret_bound_history = regret_bound_history
        return ave_rewards_history, arm_history, cumu_regret_history, regret_bound_history

    def e_greedy_choose_arm(self, epsilon):
        avg_rewards = [arm.get_ave_reward() for arm in self.arms]
        if np.random.rand() > epsilon:
            return np.argmax(avg_rewards)
        else:
            return np.random.randint(self.num_arms)

    # This epsilon greedy theoretical bound is not good. We may not use it for now.
    def e_greedy_regret_bound(self, epsilon, n, c, d):
        if epsilon == 1:
            return 1
        p = c / (d**2 * n) + 2 * c / d**2 * np.log((n-1) * d**2 * np.e**0.5 / (c * self.num_arms)) * (c * self.num_arms / ((n-1) * d**2 * np.e**0.5))**(c / (5 * d**2)) + 64 * np.e / d * (c * self.num_arms / ((n-1) * d**2 * np.e**0.5))**(c / 32)
        if self.print_progress:
            if n % 100 == 0:
                print "iteration n =", n, "p =", p
        return p * (self.num_arms - 1)

    def E_greedy(self):
        for arm in self.arms:
            arm.initialize_arm()

        cumulative_reward = 0.0
        cumulative_regret = 0.0
        cumulative_regret_bound = 0.0

        ave_rewards_history = []
        arm_history = []
        cumu_regret_history = []
        regret_bound_history = []

        # computing constants for e greedy
        tau_list = []
        for arm in self.arms:
            C, eta = arm.get_ergodic_constants()
            tau_j = 0
            for k in range(int(np.log(self.horizon)), int(np.log(self.horizon))+1000):
                L = 0.5 * C / (1 - eta) * (1 / (self.tau0 + self.a) + 1 / self.a * np.log(1 + k * self.a / self.tau0))
                tau_temp = L / np.sqrt(k)
                if tau_temp > tau_j:
                    tau_j = tau_temp
            tau_list.append(tau_j)
        c = 32 * max(tau_list)**2
        d = self.min_delta

        for n in range(self.horizon):
            epsilon = c * self.num_arms / (d**2 * (n+1))
            if self.print_progress:
                if n % 100 == 0:
                    print "iteration n =", n, "epsilon =", epsilon
            if epsilon > 1:
                epsilon = 1
            index = self.e_greedy_choose_arm(epsilon)
            iteration = int(self.tau0 + self.a * n)
            if self.stop_early:
                if iteration > self.stopping_threshold:
                    iteration = self.stopping_threshold
            reward, self.state = self.arms[index].pull(self.state, iteration)
            cumulative_reward += reward
            cumulative_regret += self.optimal_expect_reward - reward
            ave_rewards_history.append(cumulative_reward / (n + 1))
            arm_history.append(index)
            cumu_regret_history.append(cumulative_regret)
            cumulative_regret_bound += self.e_greedy_regret_bound(epsilon, n+1, c, d)
            regret_bound_history.append(cumulative_regret_bound)

        self.ave_rewards_history = ave_rewards_history
        self.arm_history = arm_history
        self.cumu_regret_history = cumu_regret_history
        self.regret_bound_history = regret_bound_history
        return ave_rewards_history, arm_history, cumu_regret_history, regret_bound_history


    def get_x(self, s, a):
        self.x = np.zeros((self.num_states*self.num_arms+1, 1))
        self.x[-1, :] = 1
        self.x[a*self.num_states:a*self.num_states+self.num_states, 0] = s

        return self.x


    def get_q(self, s, a):
        return float(self.w.T.dot(self.get_x(s, a)))


    def e_greedy_choose_arm_q_learn(self, s, epsilon):
        if not np.random.binomial(1, epsilon):
            a = np.argmax(np.array([self.get_q(s, a) for a in xrange(self.num_arms)]))
        else:
            a = np.random.choice(range(self.num_arms))
        return a


    def RL(self, gamma=.9, alpha=.2, epsilon=.2, alpha_decay_param=.00001, alpha_decay=True, 
           epsilon_decay_param=.00001, epsilon_decay=True, noisy=False):

        for arm in self.arms:
            arm.initialize_arm()

        cumulative_reward = 0.0
        cumulative_regret = 0.0

        ave_rewards_history = []
        cumu_regret_history = []
        arm_history = []

        self.w = np.zeros((self.num_states*self.num_arms+1, 1))

        if noisy:

            self.belief_arms = [Arm(self.num_states) for _ in range(self.num_arms)] 
            # This is the incorrect state we are propagating.
            self.belief_state = np.random.rand(1, self.num_states)
            self.belief_state = preprocessing.normalize(self.belief_state, norm='l1', 
                                                        axis=1, copy=True, return_norm=False)

        for n in range(self.horizon):
            if self.print_progress:
                if n % 100 == 0 and n > 0:
                    print "iteration n =", n, "reward =", ave_rewards_history[-1]
            if epsilon_decay:
                epsilon = epsilon * np.exp(-epsilon_decay_param*n)
            if alpha_decay:
                alpha = alpha * np.exp(-alpha_decay_param*n)

            if noisy:
                index = self.e_greedy_choose_arm_q_learn(self.belief_state, epsilon)
            else:
                index = self.e_greedy_choose_arm_q_learn(self.state, epsilon)

            iteration = int(self.tau0 + self.a * n)
            if self.stop_early:
                if iteration > self.stopping_threshold:
                    iteration = self.stopping_threshold

            if noisy:
                reward, self.state = self.arms[index].pull(self.state, iteration)
                _, self.belief_state = self.belief_arms[index].pull(self.belief_state, iteration)
            else:
                old = self.state
                reward, self.state = self.arms[index].pull(self.state, iteration)

            if noisy:
                self.w += alpha*(reward + gamma*max([self.get_q(self.belief_state, a_new) 
                                                    for a_new in xrange(self.num_arms)]) 
                                 - self.get_q(self.belief_state, index))*(self.get_x(self.belief_state, index))
            else:
                self.w += alpha*(reward + gamma*max([self.get_q(self.state, a_new) 
                                                    for a_new in xrange(self.num_arms)]) 
                                 - self.get_q(self.state, index))*(self.get_x(self.state, index))

            cumulative_reward += reward
            cumulative_regret += self.optimal_expect_reward - reward
            ave_rewards_history.append(cumulative_reward / (n + 1))
            arm_history.append(index)
            cumu_regret_history.append(cumulative_regret)

        self.ave_rewards_history = ave_rewards_history
        self.cumu_regret_history = cumu_regret_history
        self.arm_history = arm_history
        return ave_rewards_history, arm_history, cumu_regret_history

