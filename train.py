import numpy as np


def one_hot_encode(state, state_space):
    one_hot = np.zeros(state_space)
    if isinstance(state, tuple):
        # Extract the state from the tuple if reset() returns a tuple
        state = state[0]
    one_hot[int(state)] = 1  # Ensure state is converted to integer
    return one_hot


def train(env, agent, n_episodes):

    # Create the environment
    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

    # Instantiate the agent
    # agent = DQNAgent(env)

    win_pct_list = []
    scores = []

    # Training loop
    for i in range(n_episodes):
        state = agent.env.reset()  # Reset the environment
        state = one_hot_encode(state, env.observation_space.n)
        done = False
        score = 0
        while not done:
            hidden = agent.q_network.init_hidden(1).to(agent.device)
            # Choose action based on epsilon-greedy policy
            action, hidden = agent.choose_action(state, hidden)
            next_state, reward, done, info = agent.env.step(
                action)  # Take the action
            # print(next_state, reward, done, info)
            next_state = one_hot_encode(next_state, env.observation_space.n)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()  # Update Q-network
            state = next_state  # Move to the next state
            score += reward
        scores.append(score)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])


def train_transformer(env, agent, n_episodes):
    win_pct_list = []
    scores = []

    # Training loop
    for i in range(n_episodes):
        state = agent.env.reset()  # Reset the environment
        state = one_hot_encode(state, env.observation_space.n)
        done = False
        score = 0
        position = 0
        while not done:
            # Choose action based on epsilon-greedy policy
            action = agent.choose_action(state, position)
            next_state, reward, done, info = agent.env.step(
                action)  # Take the action
            next_state = one_hot_encode(next_state, env.observation_space.n)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()  # Update Q-network
            state = next_state  # Move to the next state
            score += reward
            position += 1
        scores.append(score)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            win_pct_list.append(avg_score)
            print('episode', i, 'win pct %.2f' %
                  avg_score, 'epsilon %.2f' % agent.epsilon)
