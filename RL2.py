import numpy as np
import MDP

class RL2:
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

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action
        '''
        nActions = self.mdp.nActions
        # calculate exponentials
        exps = [np.exp(policyParams[a, state]) for a in range(0, nActions)]
        # divide exponentials by the sum to get action probabilities
        action_prob = [i/sum(exps) for i in exps]
        # now sample an action based on those probabilities
        # need cumulative sum in order to select
        action_prob_cum = np.array(action_prob).cumsum()
        r = np.random.random()
        for n in range(0, nActions):
            if r < action_prob_cum[n]:
                action = n

        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs:
        V -- final value function
        policy -- final policy
        '''
        T, R, nStates, nActions, discount = defaultT, initialR, self.mdp.nStates, self.mdp.nActions, self.mdp.discount


        V = np.zeros(nStates)
        policy = np.zeros(nStates,int)
        n = np.zeros([nActions,nStates])
        n_ = np.zeros([nActions,nStates,nStates])
        rewards=[]

        for epi in range(nEpisodes):
            s = s0 # initial state
            total_reward = 0
            for step in range(nSteps):
                if np.random.rand(1) < epsilon:
                    # choose a as random
                    a = np.random.randint(0, nActions)
                else:
                    a = np.argmax([R[a_,s] for a_ in range(nActions)])
                [r, s_] = self.sampleRewardAndNextState(s, a)
                total_reward += r # record cumulative reward for plot
                # update counts
                n[a,s] +=  1
                n_[a,s,s_] += 1
                # update transition
                for next_state in range(nStates):
                    T[a,s,next_state] = n_[a,s,next_state] / n[a,s]
                # update reward
                R[a,s] = (r + ((n[a,s] - 1) * R[a,s]))/ n[a,s]
                # update Value
                for s in range(nStates):
                    V[s] = max(R[a,s] for a in range(nActions))  + \
                                discount * sum([p * V[s_] for (s_, p) in enumerate(T[a,s])])
                                #for a in range(nActions)]
                s = s_
            rewards.append(total_reward)
        for s in range(nStates):
            policy[s] = np.argmax([sum([p * (R[a,s] + discount * V[s_]) for (s_, p) in \
                enumerate(T[a,s])]) for a in range(nActions)])
        print(total_reward)
        #import matplotlib.pyplot as plt
        #plt.plot(rewards)
        #plt.show()
        return [V,policy]

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        R, nActions = self.mdp.R, self.mdp.nActions
        epsilon = 0.1
        total_reward = 0
        counts_a = np.zeros(self.mdp.nActions)
        empiricalMeans = np.zeros(nActions)

        for t in range(1, nIterations+1):
            reward=0
            if np.random.random() > epsilon:
                a = np.argmax(empiricalMeans)
            else:
                a = np.random.randint(nActions)
            counts_a[a] += 1
            reward = self.sampleReward(R[a])
            total_reward += reward
            empiricalMeans[a] += (reward - empiricalMeans[a]) / counts_a[a]
        print(total_reward)
        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        R, nActions = self.mdp.R, self.mdp.nActions
        total_reward = 0
        posterior = np.zeros(nActions)
        empiricalMeans = np.zeros(nActions)
        empiricalMeans_sum = np.zeros(nActions)

        for n in range(1, nIterations+1):
            reward=0
            for a in range(0, nActions):
                # sample theta k from beta(alpha, beta)
                posterior[a] = np.random.beta(prior[a,0], prior[a,1])
                empiricalMeans_sum[a] += posterior[a]
                empiricalMeans[a] = empiricalMeans_sum[a] / n

            #select best action (a* : argmax of the empirical means)
            a_ = np.argmax(empiricalMeans)

            # do best action and get reward
            reward = self.sampleReward(R[a_])

            # update total reward
            total_reward += reward

            # update distributon (prior[a*,0] += reward, beta = beta + 1 - reward)
            prior[a_,0] += reward
            prior[a_,1] += 1 - reward
        print(total_reward)
        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        R, nActions = self.mdp.R, self.mdp.nActions
        total_reward = 0
        counts_a = np.ones(nActions)
        posterior = np.zeros(nActions)
        empiricalMeans = np.zeros(nActions)
        empiricalMeans_sum = np.zeros(nActions)
        upper_bound = np.zeros(nActions)

        for n in range(1, nIterations+1):
            reward=0

            for a in range(0, nActions):
                upper_bound[a] = empiricalMeans[a] + np.sqrt((2*np.log(n))/counts_a[a])

            #select best action (a* : argmax of the empirical means)
            a_ = np.argmax(upper_bound)

            # do best action and get reward
            reward = self.sampleReward(R[a_])

            # update total reward
            total_reward += reward

            # update empiricalMeans
            empiricalMeans[a_] = (counts_a[a_]*empiricalMeans[a_] + reward) / (counts_a[a_] + 1)

            counts_a[a_] += 1
        print(total_reward)

        return empiricalMeans

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''
        R, nActions, nStates, discount = self.mdp.R, self.mdp.nActions, self.mdp.nActions, self.mdp.discount
        policyParams = initialPolicyParams
        data = []
        G = np.zeros(nSteps)

        for epi in range(0, nEpisodes):
            s = s0
            for step in range(0, nSteps):
                # do the episode
                a = self.sampleSoftmaxPolicy(policyParams, s)
                r, s_ = self.sampleRewardAndNextState(s,a)

                data.append([s,a,r])

            # update G and policy per step
            for n in range(0, nSteps):
                # update G
                G[n] = sum([discount**t * data[n+t][2] for t in range(0,nSteps-n)])
                # update policy
                # policyParams += alpha * (discount ** n) * G[n] * 0

        return policyParams
