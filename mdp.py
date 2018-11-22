import numpy as np


def prettify(reward):
    """Emphasize the sign of the specified reward
    Negative rewards are untouched, but we prepend a '+' to positive ones
    """
    if reward >= 0:
        reward = "+{}".format(reward)
    return reward


class MDP(object):
    """Class representing a Markov Decision Process, an meant
    to be used in toy examples such as betting games.
    """
    def __init__(self, seed, name, state_space, action_space, transitions, rewards, gamma=0.99):
        self.seed = seed
        np.random.seed(self.seed)
        self.name = name
        # Initialize state and action spaces
        self.state_space = state_space
        self.action_space = action_space
        # Initialize transitions
        self.transitions = transitions
        # Initialize rewards
        self.rewards = rewards
        # Initialize discount factor
        self.gamma = gamma
        # Reset (or rather, initialize) the mdp
        self.reset()

    def step(self, action):
        """Perform a step in the markov decision process"""
        assert action in self.action_space, "either not int or not admissible"
        # Sample uniformly from the unit ball
        unit_sample = np.random.uniform()
        # Get the next state and the reward defining the transition
        next_state = self.transitions[(self.current_state, action)](unit_sample)
        reward = self.rewards[(self.current_state, action)](unit_sample)
        # Update the current state
        self.current_state = next_state
        return next_state, reward

    def reset(self):
        self.current_state = 0


# Specify the entities composing the MDP
state_space = [0, 1]
action_space = [0, 1]
transitions = {(0, 0): lambda x: 0 if x < 0.6 else 1,
               (0, 1): lambda x: 0,
               (1, 0): lambda x: 0,
               (1, 1): lambda x: 0}
rewards = {(0, 0): lambda x: 2 if x < 0.6 else 0,
           (0, 1): lambda x: 3,
           (1, 0): lambda x: 0,
           (1, 1): lambda x: 1000}
# Create the MDP
mdp = MDP(1, 'toy', state_space, action_space, transitions, rewards)

while True:
    print(64 * '-')
    # The 'input' statement prevents to loop from racing
    print("You are in state: [ {} ]".format(mdp.current_state))
    action_from_human = input("Do something: ")
    if action_from_human in ['exit', 'bye', 'bb']:
        print("\n\nBye.\n")
        raise SystemExit
    next_state, reward = mdp.step(int(action_from_human))
    print("You did action: [ {} ]".format(action_from_human))
    print("\nYou arrived in state: [ {} ]".format(next_state))
    print("You received a [ {} ] reward".format(prettify(reward)))
