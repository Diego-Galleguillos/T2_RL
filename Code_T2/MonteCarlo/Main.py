from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv, CliffVisualizer
import random, time
import matplotlib.pyplot as plt


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
            Q_sa[state][action] = 0
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
    counter = 1
    results = []
    first = True
    if env_type == 0:
        plot_every = 5*10**5
    else:
        plot_every = 10**3
    for episode in range(n_episodes):
        trace = generate_episode(env, policy, actions, epsilon)
        G = 0
        #print(trace)
        for state, action, reward in trace:
            #print(f"State: {state}, Action: {action}, Reward: {reward}, G: {G}")
            G = G*discount + reward
            N_sa[state][action] += 1
            Q_sa[state][action] += (G-Q_sa[state][action])/N_sa[state][action]
            policy[state] = best_action_for_state(Q_sa, state, actions)
        if counter == plot_every or first:
            if env_type == 0:
                results.append([agent_play_blackjack(policy), episode])
            else:
                results.append([agent_play_cliff(policy), episode])
            counter = 1
            first = False
        else:
            counter += 1
    #plot result
    if env_type == 0:
        title = "Blackjack"
    else:
        title = "Cliff"

    returns, episodes = zip(*results)
    plt.plot(episodes, returns, marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('Monte Carlo:' + title)
    plt.grid(True)
    plt.show()


    return policy

def agent_play_blackjack(policy):
    env = BlackjackEnv()
    avg_return = 0
    for i in range(10**5):
        actions = env.action_space
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = policy[state]#random.choice(actions)
            state, reward, done = env.step(action)
            total_reward += reward
        avg_return += total_reward/10**5
    return avg_return

def agent_play_cliff(policy, cliff_width=6):
    env = CliffEnv(cliff_width)
    avg_return = 0
    for i in range(1):
        actions = env.action_space
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = policy[state]#random.choice(actions)
            state, reward, done = env.step(action)
            total_reward += reward
        avg_return += total_reward/1
    return avg_return
    
def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)

def pregunta_j_blackjack():
    for i in range(5):
        policy = MC_control_every_visit(0, 0.01, 1, 10**7)
        
def pregunta_j_cliff():
    for i in range(5):
        policy = MC_control_every_visit(1, 0.1, 1, 2*10**5)

if __name__ == '__main__':
    #play_blackjack()
    #play_cliff()

    pregunta_j_cliff()

