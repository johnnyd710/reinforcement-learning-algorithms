import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount

    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        T, R, discount, nStates, nActions = self.T, self.R, self.discount, self.nStates, self.nActions
        V = initialV.copy()

        iterId = 0
        while True:
            V_ = V.copy()
            # stopping condition
            epsilon=0
            for s in range(nStates):
                V[s] = max(R[a,s] for a in range(nActions))  + \
                            discount * max([sum([p * V_[s_] for (s_, p) in enumerate(T[a,s])]) \
                            for a in range(nActions)])
                epsilon = max(epsilon, abs(V[s] - V_[s]))
            iterId = iterId + 1
            if epsilon <= tolerance or iterId >= nIterations:
                #print("Success, number of iterations: %d \n Values: %s" % (iterId, V))
                break



        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        nStates, T, R, discount, nActions = self.nStates, self.T, self.R, self.discount, self.nActions
        policy = np.zeros(nStates)

        for s in range(nStates):
            policy[s] = np.argmax([sum([p * (R[a,s] + discount * V[s_]) for (s_, p) in \
            enumerate(T[a,s])]) for a in range(nActions)])

        #print("Optimal Policies: %s" % policy)
        return policy

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''


        T, R, discount, nStates, nActions = self.T, self.R, self.discount, self.nStates, self.nActions
        V = np.zeros(nStates)
        P = np.zeros((nStates,nStates))
        for (s, a) in enumerate(policy):
            P[s] = T[a, s]

        # Solve Linear System: A x = R
        A = (np.identity(nStates) - discount*P)
        V = np.linalg.solve(A, R[0])

        #print(V)
        return V

    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        T, R, discount, nStates, nActions = self.T, self.R, self.discount, self.nStates, self.nActions
        V = np.zeros(nStates)
        policy = initialPolicy
        iterId = 0

        #print("Initial Policy = %s" % policy)

        while True:
            noChange = True
            iterId = iterId + 1
            # policy evaluation
            V = self.evaluatePolicy(policy)

            # policy improvement
            for s in range(nStates):
                #Q_best = V[s]
                #for a in range(nActions):
                action_values = [sum([p * (R[a,s] + discount * V[s_]) for (s_, p) in \
                enumerate(T[a,s])]) for a in range(nActions)]
                #print(action_values)
                a_ = np.argmax(action_values)
                    #print("Q: %f, Q best : %f" % (Q, Q_best))
                if policy[s] != a_:
                    noChange = False
                    policy[s] = a_
                    #if Q.astype(np.int32) > Q_best.astype(np.int32):
                    #    policy[s] = a
                #        Q_best = Q
                #        noChange = False
            #print(policy)
            if noChange: break
        #print("Optimal Policies: %s \n Iterations: %d" % (policy, iterId))
        return [policy,V,iterId]

    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        T, R, discount, nStates, nActions = self.T, self.R, self.discount, self.nStates, self.nActions

        V = initialV
        iterId = 0
        epsilon = 0.0

        #print("\nEvaluate Function, policy = %s" % policy)

        while True:
            epsilon = 0
            iterId = iterId + 1
            for (s, a) in enumerate(policy):
                V_ = 0
                V_ = R[a,s] + discount * sum([p * V[s_] for (s_, p) in enumerate(T[a,s])])
                #print(V_, V[s])CS 885 assignment 1
                epsilon = max(epsilon, np.abs(V_ - V[s]))
                V[s] = V_
            #print(epsilon)
            if epsilon <= tolerance or iterId >= nIterations:
                break

        #print("Successful partial policy eval, Iterations: %d, V = %s" % (iterId, V))
        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs:
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        T, R, discount, nStates, nActions = self.T, self.R, self.discount, self.nStates, self.nActions

        policy = initialPolicy
        V = initialV
        iterId = 0

        #print("Initial Policy for modified = %s" % policy)

        while True:
            epsilon = 0.0
            iterId = iterId + 1
            # policy evaluation
            V, tmp, tmp2 = self.evaluatePolicyPartially(policy, V, nIterations=nEvalIterations)
            # policy improvement
            for s in range(nStates):
                #Q_best = V[s]
                #for a in range(nActions):
                #print(V)
                action_values = [sum([p * (R[a,s] + discount * V[s_]) for (s_, p) in \
                enumerate(T[a,s])]) for a in range(nActions)]
                V_ = np.max(action_values)
                policy[s] = np.argmax(action_values)

                epsilon = max(epsilon, np.abs(V_ - V[s]))
                V[s] = V_

            if (epsilon < tolerance):
                break
            #print(V)
                #if policy[s] != a:
                #    noChange = False
            #        policy[s] = a_

                    #if Q.astype(np.int32) > Q_best.astype(np.int32):
                    #    policy[s] = a
                #        Q_best = Q
                #        noChange = False
            #print(policy)
            #if noChange: break
        #print("Optimal Policies: %s \n Value %s \n Iterations: %d" % (policy, V, iterId))

        return [policy,V,iterId,epsilon]
