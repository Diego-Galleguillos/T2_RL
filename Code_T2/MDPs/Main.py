import random, numpy, time
import matplotlib.pyplot as plt 

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        print(state)
        print(type(state))
        state = s_next
        print(state)
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")


def iterative_pol_general(param, discount, problem_type):
    V = {}
    if problem_type == 0:
        problem = GridProblem(param)
    elif problem_type == 1:
        problem = CookieProblem(param)
    else:
        problem = GamblerProblem(param)
    ini_state = problem.get_initial_state()
    states = get_states(problem_type, param)
    for state in states:
        V[state] = 0
    theta = 1
    counter = 0
    while theta > 0.0000000001:
        theta = 0
        counter += 1
        for state in states:
            skip = problem.is_terminal(state)
            if not skip:
                v = V[state]
                actions = problem.get_available_actions(state)
                sum_actions = 0
                for action in actions:
                    transitions = problem.get_transitions(state, action)
                    sum_dynamics = 0
                    for prob, s_next, reward in transitions:
                        sum_dynamics += prob * (reward + discount * V[s_next])
                    sum_actions += (1 / (len(actions))) * sum_dynamics
                V[state] = sum_actions
                theta = max(theta, abs(V[state] - v))
    return V, problem_type, counter, param, ini_state

def value_iteration(param, discount, problem_type):
    V = {}
    policy = {}
    if problem_type == 0:
        problem = GridProblem(param)
    elif problem_type == 1:
        problem = CookieProblem(param)
    else:
        problem = GamblerProblem(param)
    states = get_states(problem_type, param)
    ini_state = problem.get_initial_state()
    for state in states:
        V[state] = 0
        policy[state] = None
    theta = 1
    counter = 0
    while theta > 0.0000000001:
        theta = 0
        counter += 1
        for state in states:
            skip = problem.is_terminal(state)
            if not skip:
                v = V[state]
                actions = problem.get_available_actions(state)
                action_results = {}
                max_V = float('-inf')
                for action in actions:
                    transitions = problem.get_transitions(state, action)
                    sum_dynamics = 0
                    for prob, s_next, reward in transitions:
                        sum_dynamics += prob * (reward + discount * V[s_next])
                    action_results[action] = sum_dynamics
                    if sum_dynamics > max_V:
                        max_V = sum_dynamics
                policy[state] = argmax(action_results)
                V[state] = max_V
                theta = max(theta, abs(V[state] - v))
    return V, problem_type, counter, param, ini_state


def get_states(problem, param):
    states = []
    if problem == 0:
        for i in range(param*param):
            states.append(i)
    elif problem == 1:
        for i in range(param*param):
            for j in range(param*param):
                    states.append((i,j))
    else:
        for i in range(101):
            states.append(i)
    return states

def show_value_function_matrix(V, problem_type, counter, param):
    if problem_type == 0:
        value_matrix = numpy.reshape([V[i] for i in range(param*param)], (param, param))
        print("Value Function Matrix:")
        for row in value_matrix:
            print(row)
    elif problem_type == 1:
        for i in range(param*param):
            matrix = numpy.zeros((param, param))
            for j in range(param*param):
                row = j // param
                col = j % param
                matrix[row][col] = V[(i, j)]
            print(f"Para posicion {i} de la galleta:")
            for row in matrix:
                print(row)
    else:
        print("Value Function Array:")
        value_array = [V[i] for i in range(101)]
        print(value_array)
    print(counter)

def argmax(d):
    max_key = None
    max_value = float('-inf')
    for key, value in d.items():
        if value > max_value:
            max_value = value
            max_key = key
    return max_key

def greedy_policy(param, discount, problem_type , V):
    policy = {}
    if problem_type == 0:
        problem = GridProblem(param)
    elif problem_type == 1:
        problem = CookieProblem(param)
    else:
        problem = GamblerProblem(param)
    states = get_states(problem_type, param)
    for state in states:
        skip = problem.is_terminal(state)
        if not skip:
            actions = problem.get_available_actions(state)
            action_values = {}
            for action in actions:
                transitions = problem.get_transitions(state, action)
                sum_dynamics = 0
                for prob, s_next, reward in transitions:
                    sum_dynamics += prob * (reward + discount * V[s_next])
                action_values[action] = sum_dynamics
            best_action = argmax(action_values)
            policy[state] = best_action
    return policy

def show_policy(policy, problem_type, param):
    if problem_type == 0:
        policy_matrix = numpy.empty((param, param), dtype=object)
        for state, action in policy.items():
            row = state // param
            col = state % param
            policy_matrix[row][col] = action
        print("Policy Matrix:")
        for row in policy_matrix:
            print(row)
    elif problem_type == 1:
        for i in range(param*param):
            matrix = numpy.empty((param, param), dtype=object)
            for j in range(param*param):
                row = j // param
                col = j % param
                if (i, j) in policy:
                    matrix[row][col] = policy[(j, i)]
                else:
                    matrix[row][col] = None
            print(f"Policy for cookie position {i}:")
            for row in matrix:
                print(row)
    else:
        print("Policy Array:")
        policy_array = [policy.get(i, None) for i in range(101)]
        print(policy_array)

def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(size)
    play(problem)


def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)

def pregunta_d():
    grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    discount = 1
    for size in grid_sizes:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = iterative_pol_general(size, discount, 0)
        end_time = time.time()
        print(f"Grid size: {size}x{size}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")

    cookie_grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    discount = 0.99
    for size in grid_sizes:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = iterative_pol_general(size, discount, 1)
        end_time = time.time()
        print(f"Cookie Grid size: {size}x{size}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")

    param_prob = [0.25, 0.4, 0.55]
    discount = 1
    for param in param_prob:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = iterative_pol_general(param, discount, 2)
        end_time = time.time()
        print(f"Prob: {param}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")

def pregunta_h():
    grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    discount = 1
    for size in grid_sizes:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = value_iteration(size, discount, 0)
        end_time = time.time()
        print(f"Grid size: {size}x{size}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")
    cookie_grid_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    discount = 0.99
    for size in grid_sizes:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = value_iteration(size, discount, 1)
        end_time = time.time()
        print(f"Cookie Grid size: {size}x{size}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")
    param_prob = [0.25, 0.4, 0.55]
    discount = 1
    for param in param_prob:
        start_time = time.time()
        V, problem_type, counter, param, ini_state = value_iteration(param, discount, 2)
        end_time = time.time()
        print(f"Prob: {param}, Time taken: {end_time - start_time:.3f} seconds, value on inicial state: {V[ini_state]:.3f}")

def pregunta_g():
    pass 

def pregunta_i():
    param_prob = [0.25, 0.4, 0.55]
    discount = 1
    for param in param_prob:
        problem = GamblerProblem(param)
        start_time = time.time()
        V, problem_type, counter, param, ini_state = value_iteration(param, discount, 2)
        end_time = time.time()
        states = get_states(2, param)
        optimal_actions = {}
        for state in states:
            skip = problem.is_terminal(state)
            if not skip:
                optimal_actions_state = []
                actions = problem.get_available_actions(state)
                q_sa = {}
                for action in actions:
                    transitions = problem.get_transitions(state, action)
                    value_action_state = 0
                    for prob, s_next, reward in transitions:
                        value_action_state += prob * (reward + discount * V[s_next])
                    q_sa[action] = round(value_action_state, 5)
                max_q = max(q_sa.values())
                for action, value in q_sa.items():
                    if value == max_q:
                        optimal_actions_state.append(action)
                optimal_actions[state] = optimal_actions_state
        x_plot = []
        y_plot = []
        for state, actions in optimal_actions.items():
            for action in actions:
                x_plot.append(state)
                y_plot.append(action)

        plt.figure()
        plt.scatter(x_plot, y_plot, s=10)
        #arreglar titutlo
        plt.title(f'Optimal Actions  (p={param})')
        plt.xlabel('State')
        plt.ylabel('Optimal Action')
        plt.grid(True)
        plt.show()        

            

        





if __name__ == '__main__':
    pregunta_h()