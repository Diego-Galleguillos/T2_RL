from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
import random


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        print(state)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")


def get_states(env_type, param=None):
    states = []
    if env_type == 0:
        for i in range(12, 22):
            for j in range(2):
                for k in range(1, 12):
                    states.append((i, bool(j), k))
    else:
        grid_size = param if param is not None else 6
        for i in range(grid_size):
            for j in range(grid_size):
                states.append((i,j))
    
    return states


def initialize_env(env_type, param=None):
    if env_type == 0:
        env = BlackjackEnv()
    else:
        cliff_width = param if param is not None else 6
        env = CliffEnv(cliff_width)
    policy = {}
    Q_sa = {}
    N_sa = {}
    states = get_states(env_type, param)
    actions = env.action_space
    for state in states:
        policy[state] = actions[0]
        Q_sa[state] = {}
        N_sa[state] = {}
        for action in actions:
            Q_sa[state][action] = 0.0
            N_sa[state][action] = 0.0
    return env, states, policy, Q_sa, N_sa, actions

def generate_episode(env, policy, actions, epsilon):
    state = env.reset()
    done = False
    episode_trace = []
    while not done:
        original_state = state
        if random.random() > epsilon:
            action = policy[state]
        else:
            action = random.choice(actions)
        state, reward, done = env.step(action)
        episode_trace.append([original_state, action, reward])
    return episode_trace[::-1]

def argmax(d):
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key

def best_action_for_state(Q_sa, state, actions):
    action_values = {}
    for action in actions:
        action_values[action] = Q_sa[state][action]
    best_action = argmax(action_values)
    return best_action



def MC_control_every_visit(env_type, epsilon, discount, n_episodes, param=None):
    env, states, policy, Q_sa, N_sa, actions = initialize_env(env_type)
    for episode in range(n_episodes):
        trace = generate_episode(env, policy, actions, epsilon)
        G = 0
        print(trace)
        for state, action, reward in trace:
            G = G*discount + reward
            N_sa[state][action] += 1
            Q_sa[state][action] += (G-Q_sa[state][action])/N_sa[state][action]
            policy[state] = best_action_for_state(Q_sa, state, actions)
    print(Q_sa)
    return policy

def agent_play_blackjack(policy):
    env = BlackjackEnv()
    performance = 0
    tie_pcg = 0
    for i in range(10000000):
        actions = env.action_space
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = policy[state]#random.choice(actions)
            print(state)
            state, reward, done = env.step(action)
            total_reward += reward
        env.show()
        print("Done.")
        if total_reward == -1:
            total_reward = 0
        elif total_reward == 0:
            tie_pcg += 1
        performance += total_reward
    print(f"Total reward: {performance/100000}%")
    print(f"Tie pcg: {tie_pcg/1000}%")


    
def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


if __name__ == '__main__':
    #play_blackjack()
    #play_cliff()
    policy = MC_control_every_visit(1, 0.5, 1, 100)
    print(policy)
    #agent_play_blackjack(policy)
#-3927.0