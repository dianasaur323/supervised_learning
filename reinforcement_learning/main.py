import gym
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def basic_plot(title, x, xlabel, y, ylabel, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'o-', color="g")
    plt.savefig(filename)
    return

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function

    V = np.zeros(env.env.nS)
    while True:
        # TODO: Implement!
        delta = 0  #delta = change in value of state from one iteration to next

        for state in range(env.env.nS):  #for all states
            val = 0  #initiate value as 0

            for action,act_prob in enumerate(policy[state]): #for all actions/action probabilities
                for prob,next_state,reward,done in env.env.P[state][action]:  #transition probabilities,state,rewards of each action
                    val += act_prob * prob * (reward + discount_factor * V[next_state])  #eqn to calculate
            delta = max(delta, np.abs(val-V[state]))
            V[state] = val
        if delta < theta:  #break if the change in value is less than the threshold (theta)
            break
    return np.array(V)

def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.99):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    def one_step_lookahead(state, V, nS, nA):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    iterations = 0
    start_time = time.time()
    values = []
    changes = []

    while True:
        # Implement this!
        old_policy = policy
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  #eval current policy
        policy_stable = True  #Check if policy did improve (Set it as True first)
        value = 0
        for state in range(env.env.nS):  #for each states
            # env.render()
            chosen_act = np.argmax(policy[state])  #best action (Highest prob) under current policy
            act_values = one_step_lookahead(state,curr_pol_val, env.env.nS, env.env.nA)  #use one step lookahead to find action values
            best_act = np.argmax(act_values) #find best action
            value += np.argmax(act_values)
            if chosen_act != best_act:
                policy_stable = False  #Greedily find best action
            policy[state] = np.eye(env.env.nA)[best_act]  #update
        iterations += 1
        values.append(value)
        changes.append(np.count_nonzero(policy!=old_policy))
        # basic_plot("Value Iteration - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes")
        # basic_plot("Value Iteration - Value", list(range(len(values))), "Iterations", values, "Values", "values")
        if policy_stable:
            print(iterations)
            basic_plot("Policy Iteration", list(range(len(values))), "Iterations", values, "Values","values_policy")
            basic_plot("Policy Iteration", list(range(len(changes))), "Iterations", changes, "Changes","changes_policy")
            print("--- %s seconds ---" % (time.time() - start_time))
            return policy, curr_pol_val

    return policy, np.zeros(nS)

def value_iteration(env, theta=0.0001, discount_factor=0.99):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for act in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][act]:
                A[act] += prob * (reward + discount_factor*V[next_state])
        return A

    def check_current_policy(state, V):
        policy = np.zeros([env.env.nS, env.env.nA])
        for state in range(env.env.nS):  #for all states, create deterministic policy
            act_val = one_step_lookahead(state,V)
            best_action = np.argmax(act_val)
            policy[state][best_action] = 1
        return policy

    V = np.zeros(env.env.nS)
    series = []
    changes = []
    values = []
    policy = np.zeros([env.env.nS, env.env.nA])
    start_time = time.time()
    while True:
        delta = 0  #checker for improvements across states
        value = 0
        for state in range(env.env.nS):
            act_values = one_step_lookahead(state,V)  #lookahead one step
            best_act_value = np.max(act_values) #get best action value
            delta = max(delta,np.abs(best_act_value - V[state]))  #find max delta across all states
            V[state] = best_act_value  #update value to best action value
            value += V[state]
        if delta < theta:  #if max improvement less than threshold
            break
        old_policy = policy
        policy = check_current_policy(state,V)
        changes.append(np.count_nonzero(policy!=old_policy))
        series.append(delta)
        values.append(value)
    policy = np.zeros([env.env.nS, env.env.nA])
    for state in range(env.env.nS):  #for all states, create deterministic policy
        act_val = one_step_lookahead(state,V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1
    print("--- %s seconds ---" % (time.time() - start_time))
    print(len(series))
    print(changes)
    basic_plot("Value Iteration - Delta", list(range(len(series))), "Iterations", series, "Delta","delta")
    basic_plot("Value Iteration - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes")
    basic_plot("Value Iteration - Value", list(range(len(values))), "Iterations", values, "Values", "values")

    # Implement!
    return policy, V

def q_learning(env, alpha = 0.7, gamma = 0.9, theta = 0.01, epsilon = 0.99, epsilon_decay = 0.9999):
    nS=env.env.nS
    nA=env.env.nA
    Q = np.zeros((nS, nA),dtype=np.float64)
    changes = []
    values = []
    policy = np.argmax(Q, axis=1 )
    start_time = time.time()
    for i in range(100000):
        oldQ = Q
        old_policy = policy
        state = env.reset() #start stepping through an episode
        while True:
            action = env.action_space.sample() \
            if np.random.random() < epsilon else np.argmax(Q[state])
            nstate, reward, done, info = env.step(action)
            Q[state][action] += alpha * (reward + gamma * Q[nstate].max() * (not done) - Q[state][action])
            state = nstate
            # epsilon *= epsilon_decay
            if done:
                break
        # print ("iteration: ",i)
        epsilon *= epsilon_decay
        policy = np.argmax(Q, axis=1 )
        changes.append(np.count_nonzero(policy!=old_policy))
        values.append(np.sum(Q))
        if abs(np.sum(Q) - np.sum(oldQ)) < theta and i > 60000:
            print("iteration: ",i)
            break
        # print("epsilon:",epsilon)
    pi = np.argmax(Q, axis=1 )
    print("--- %s seconds ---" % (time.time() - start_time))
    return pi, Q, changes, values

environment = input("Model name?: ").upper()
if(environment == 'T'):
    env = gym.make('Taxi-v2')
    # print("----- VALUE ITERATION -----")
    # random_policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    # Compare different policies
    # print("----- discount factor = 0.9, theta = 0.0001 -----")
    # policy1 = value_iteration(env, theta = 0.0001, discount_factor = 0.9)[0]
    # print("----- discount factor = 0.99, theta = 0.0001 -----")
    # policy2 = value_iteration(env, theta = 0.0001, discount_factor = 0.99)[0]
    # print("----- discount factor = 0.999, theta = 0.0001 -----")
    # policy3 = value_iteration(env, theta = 0.0001, discount_factor = 0.999)[0]
    # print("----- discount factor 0.9 vs 0.99 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- discount factor 0.99 vs 0.999 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- discount factor 0.9 vs 0.999 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # print("----- discount factor = 0.9, theta = 0.01 -----")
    # policy1 = value_iteration(env, theta = 0.01, discount_factor = 0.9)[0]
    # print("----- discount factor = 0.9, theta = 0.001 -----")
    # policy2 = value_iteration(env, theta = 0.001, discount_factor = 0.9)[0]
    # print("----- discount factor = 0.9, theta = 0.0001 -----")
    # policy3 = value_iteration(env, theta = 0.01, discount_factor = 0.9)[0]
    # print("----- theta 0.01 vs 0.001 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- theta 0.001 vs 0.0001 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- theta 0.01 vs 0.0001 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # print("----- Actual Value Iteration -----")
    value_policy = value_iteration(env, theta = 0.01, discount_factor = 0.9)[0]
    # print("----- POLICY ITERATION -----")
    # policy1 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.9)[0]
    # policy2 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.99)[0]
    # policy3 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.999)[0]
    # print("----- discount factor 0.9 vs 0.99 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- discount factor 0.99 vs 0.999 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- discount factor 0.9 vs 0.999 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # print("----- Actual Policy Iteration -----")
    # policy_policy = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.9)[0]
    # print("----- value iteration vs policy iteration -----")
    # print(np.count_nonzero(value_policy!=policy_policy))
    print("----- Q-LEARNING -----")
    # policy1, Q, changes, values = q_learning(env, alpha = 0.7, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p1")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p1")
    # policy2, Q, changes, values  = q_learning(env, alpha = 0.8, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p2")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p2")
    # policy3, Q, changes, values  = q_learning(env, alpha = 0.9, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p3")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p3")
    # print("----- alpha 0.7 vs 0.8 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- alpha 0.8 vs 0.9 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- alpha 0.7 vs 0.9 -----")
    # print(np.count_nonzero(policy1!=policy3))
    policy1, Q, changes, values = q_learning(env, alpha = 0.9, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_gamma")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_gamma")
    policy5, Q, changes, values = q_learning(env, alpha = 0.9, gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.9999)
    basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_epsilon")
    basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_epsilon")
    print("----- epsilon 0.99 vs 0.9 -----")
    print(np.count_nonzero(policy1!=policy5))
    print("----- value iteration vs q-learning -----")
    print(np.count_nonzero(value_policy!=policy5))
else:
    env = gym.make('FrozenLake-v0')
    # print("----- VALUE ITERATION -----")
    # print("----- discount factor = 0.9, theta = 0.0001 -----")
    # policy1 = value_iteration(env, theta = 0.0001, discount_factor = 0.9)[0]
    # print("----- discount factor = 0.99, theta = 0.0001 -----")
    # policy2 = value_iteration(env, theta = 0.0001, discount_factor = 0.99)[0]
    # print("----- discount factor = 0.999, theta = 0.0001 -----")
    # policy3 = value_iteration(env, theta = 0.0001, discount_factor = 0.999)[0]
    # print("----- discount factor 0.9 vs 0.99 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- discount factor 0.99 vs 0.999 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- discount factor 0.9 vs 0.999 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # print("----- discount factor = 0.99, theta = 0.01 -----")
    # policy1 = value_iteration(env, theta = 0.01, discount_factor = 0.99)[0]
    # print("----- discount factor = 0.99, theta = 0.001 -----")
    # policy2 = value_iteration(env, theta = 0.001, discount_factor = 0.99)[0]
    # print("----- discount factor = 0.99, theta = 0.0001 -----")
    # policy3 = value_iteration(env, theta = 0.0001, discount_factor = 0.99)[0]
    # print("----- theta 0.01 vs 0.001 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- theta 0.001 vs 0.0001 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- theta 0.01 vs 0.0001 -----")
    # print(np.count_nonzero(policy1!=policy3))
    print("----- Actual Value Iteration -----")
    value_policy = value_iteration(env, theta = 0.001, discount_factor = 0.99)[0]
    # print("----- POLICY ITERATION -----")
    # policy1 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.9)[0]
    # policy2 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.99)[0]
    # policy3 = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.999)[0]
    # print("----- discount factor 0.9 vs 0.99 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- discount factor 0.99 vs 0.999 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- discount factor 0.9 vs 0.999 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # print("----- Actual Policy Iteration -----")
    # policy_policy = policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=0.99)[0]
    # print("----- value iteration vs policy iteration -----")
    # print(np.count_nonzero(value_policy!=policy_policy))
    print("----- Q-LEARNING -----")
    policy1, Q, changes, values = q_learning(env, alpha = 0.7, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p1")
    basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p1")
    # policy2, Q, changes, values  = q_learning(env, alpha = 0.8, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p2")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p2")
    # policy3, Q, changes, values  = q_learning(env, alpha = 0.9, gamma = 0.9, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_p3")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_p3")
    # print("----- alpha 0.7 vs 0.8 -----")
    # print(np.count_nonzero(policy1!=policy2))
    # print("----- alpha 0.8 vs 0.9 -----")
    # print(np.count_nonzero(policy2!=policy3))
    # print("----- alpha 0.7 vs 0.9 -----")
    # print(np.count_nonzero(policy1!=policy3))
    # policy4, Q, changes, values = q_learning(env, alpha = 0.7, gamma = 0.99, epsilon = 0.99, epsilon_decay = 0.9999)
    # basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_gamma")
    # basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_gamma")
    policy5, Q, changes, values = q_learning(env, alpha = 0.7, gamma = 0.9, epsilon = 0.9, epsilon_decay = 0.9999)
    basic_plot("Q-Learning - Policy Changes Count", list(range(len(changes))), "Iterations", changes, "Changes", "changes_epsilon")
    basic_plot("Q-Learning - Values", list(range(len(values))), "Iterations", values, "Values", "values_epsilon")
    print("----- epsilon 0.99 vs 0.9 -----")
    print(np.count_nonzero(policy1!=policy5))
    print("----- value iteration vs q-learning -----")
    print(np.count_nonzero(value_policy!=policy1))
