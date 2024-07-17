from environments import DelaySampleToMatchEnv
from agents import DQNAgent
import numpy as np
import torch
import torch.nn.functional as F
import time


def vanilla_train(config_dict):
    start = time.time()
    n_stimuli = config_dict['state_size'] - 1
    env = DelaySampleToMatchEnv(n_stimuli=n_stimuli)
    agent = DQNAgent(config_dict)

    n_episodes = config_dict['train_steps']
    n_episodes = 4000
    episode_rewards = []
    scores = []

    # Training loop
    for i in range(n_episodes):
        state = env.reset()  # Reset the environment
        state = F.one_hot(torch.tensor(state),
                          env.observation_space.n).to(dtype=torch.float32, device=agent.device)
        done = False
        score = 0
        hidden = agent.q_network.init_hidden().to(agent.device)
        while not done:
            action, next_hidden = agent.select_action(state, hidden)
            next_state, reward, done, info = env.step(
                action)  # Take the action
            next_state = F.one_hot(torch.tensor(next_state),
                                   env.observation_space.n).to(dtype=torch.float32, device=agent.device)
            # ('state', 'action', 'next_state', 'reward', 'hidden', 'next_hidden', 'done'))
            agent.store_transition(state, action, next_state,
                                   reward, hidden, next_hidden, done)
            agent.learn()  # Update Q-network
            hidden = next_hidden
            state = next_state  # Move to the next state
            score += reward

        scores.append(score)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            episode_rewards.append(avg_score)
            print(f"Episode {i} - Average Score: {avg_score:.2f}")

    end = time.time()
    agent.save()
    print(f'It took {end-start} seconds to train the model')


def cifar_train(env, agent):
    pass


def mnist_train(env, agent):
    pass
