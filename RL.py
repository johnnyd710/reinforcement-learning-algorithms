import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with
        probabilty epsilon and performing Boltzmann exploration otherwise.
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs:
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        T, R, discount, nStates, nActions = self.mdp.T, self.mdp.R, self.mdp.discount, self.mdp.nStates, self.mdp.nActions
        Q = np.zeros([nActions,nStates])
        policy = np.zeros(nStates,int)
        Q = initialQ
        n = np.zeros([nActions,nStates])
        rewards = []
        for episode in range(nEpisodes):
            s = s0
            cum_r = 0
            for step in range(nSteps):
                if np.random.rand(1) < epsilon:
                    # choose a as random
                    a = np.random.randint(0, nActions)

                elif temperature:
                    # boltzmann
                    action_prob_tmp = []
                    denom = 0
                    for a in range(nActions):
                        val = np.exp(Q[a,s]/temperature)
                        action_prob_tmp.append(val)
                        denom += val
                    action_prob = [num / denom for num in action_prob_tmp]
                    action_prob_cum = np.cumsum(action_prob)
                    rand_val = np.random.rand()
                    a = np.where(action_prob_cum >= np.random.rand())

                else:
                    a = np.argmax([Q[a_,s] for a_ in range(nActions)])

                [r, s_] = self.sampleRewardAndNextState(s, a)
                cum_r += r # record cumulative reward for plot
                # update counts
                n[a,s] = n[a,s] + 1
                alpha = 1.0/n[a,s]
                # update Q value
                Q[a,s] = Q[a,s] + alpha*(r + discount*np.max([Q[a_, s_] for \
                            a_ in range(nActions)] - Q[a,s]))
                s = s_
            rewards.append(cum_r)
        for state in range(nStates):
            policy[state] = np.argmax([Q[action,state] for action in range(nActions)])


        return [Q,policy, rewards]
