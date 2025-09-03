import random, numpy, time

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
            else:
                print(state)
    return V, problem_type, counter, param

def value_iteration(param, discount, problem_type):
    V = {}
    policy = {}
    offset = 0 #explicar o preguntar jueves
    if problem_type == 0:
        problem = GridProblem(param)
        offset = 1
    elif problem_type == 1:
        problem = CookieProblem(param)
    else:
        problem = GamblerProblem(param)
    states = get_states(problem_type, param)
    for state in states:
        V[state] = 0 + param*param*offset
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
                max_V = 0
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
    show_policy(policy, problem_type, param)

    if offset == 1:
        for state, value in V.items():
            V[state] = value-param*param*offset
    return V, problem_type, counter, param


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


if __name__ == '__main__':
    param = 0.4
    discount = 1
    problem_type = 2
    V, prob, counter, param = value_iteration(param, discount, problem_type)
    show_value_function_matrix(V, prob, counter, param )
    #greedy_policy_value = greedy_policy(param, discount, problem_type, V)
    #print(greedy_policy_value)
    #show_policy(greedy_policy_value, prob, param)
    #show_value_function_matrix(V, prob, counter, param )



'''
def iterative_pol_grid(size, discount):
    problem = GridProblem(size)
    V = numpy.zeros(size*size)
    state = problem.get_initial_state()
    theta = 1
    counter = 0
    while theta > 0.0000000001:
        theta = 0
        counter += 1
        for i in range(size*size):
            skip = problem.is_terminal(i)
            if not skip:
                state = i
                v = V[i]
                actions = problem.get_available_actions(state)
                sum_actions = 0
                for action in actions:
                    transitions = problem.get_transitions(state, action)
                    sum_dynamics = 0
                    for prob, s_next, reward in transitions:
                        sum_dynamics += prob*(reward+discount*V[s_next])
                    sum_actions += (1/(len(actions)))*sum_dynamics
                V[i] = sum_actions
                theta = max(theta, abs(V[i]-v))

    value_matrix = numpy.reshape(V, (size, size))
    print("Value Function Matrix:")
    for row in value_matrix:
        print(row)
    print(counter)

def iterative_pol_cookie(size, discount):
    problem = CookieProblem(size)
    V = numpy.zeros((size*size,size*size))
    theta = 1
    counter = 0
    while theta > 0.0000000001:
        theta = 0
        counter += 1
        for i in range(size*size):
            for j in range(size*size):
                skip = False
                if not skip:
                    state = (i,j)
                    v = V[i][j]
                    actions = problem.get_available_actions(state)
                    sum_actions = 0
                    for action in actions:
                        transitions = problem.get_transitions(state, action)
                        sum_dynamics = 0
                        for prob, s_next, reward in transitions:
                            sum_dynamics += prob*(reward+discount*V[s_next[0]][s_next[1]])
                        sum_actions += (1/(len(actions)))*sum_dynamics
                    V[i][j] = sum_actions
                    theta = max(theta, abs(V[i][j]-v))


def iterative_pol_gamble(p, discount):
    problem = GamblerProblem(p)
    V = numpy.zeros((101))
    theta = 1
    counter = 0
    while theta > 0.0000000001:
        theta = 0
        counter += 1
        for i in range(100):
            skip = i == 100 or i == 0
            if not skip:
                state = i
                v = V[i]
                actions = problem.get_available_actions(state)
                sum_actions = 0
                for action in actions:
                    transitions = problem.get_transitions(state, action)
                    sum_dynamics = 0
                    for prob, s_next, reward in transitions:
                        sum_dynamics += prob*(reward+discount*V[s_next])
                    sum_actions += (1/(len(actions)))*sum_dynamics
                V[i] = sum_actions
                theta = max(theta, abs(V[i]-v))




'''