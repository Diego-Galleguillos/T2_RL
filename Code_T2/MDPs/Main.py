import random, numpy

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
        state = s_next
        print(state)
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")

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
    #play_grid_problem()
    play_cookie_problem()
    #play_gambler_problem()
    #iterative_pol_grid(10, 1)