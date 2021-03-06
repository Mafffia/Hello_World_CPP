import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
        从重放存储器中对batch_size转换进行采样，并返回一个元组
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        

        super(DQN, self).__init__()
        
        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        # LY add
        self.step = 0

        self.fc1 = nn.Linear(4, 256).to(device)
        self.fc1
        self.fc2 = nn.Linear(256, self.n_actions).to(device)
        self.fc2
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        
        x = self.relu(self.fc1(x)).to(device)
        x = self.fc2(x).to(device)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        if exploit is False:
            self.step += 1
            if self.step >= self.anneal_length:
                eps = self.eps_end
            else:
                eps = self.eps_start - self.step * (self.eps_start - self.eps_end) / self.anneal_length

            shape = (len(observation), 1)
            action = torch.ones(shape)
            q = self.forward(observation)
            q = q.detach().cpu().numpy()
            for i in range(len(observation)):
                if np.random.random_sample() <= eps:
                    action[i][0] = int(np.random.choice(range(self.n_actions)))
                else:
                    action[i][0] = int(np.nanargmax(q[i, :], axis=0))
        else:
            shape = (len(observation), 1)
            action = torch.ones(shape)
            q = self.forward(observation)
            q = q.detach().cpu().numpy()
            for i in range(len(observation)):
                action[i][0] = int(np.nanargmax(q[i, :], axis=0))
        return action

        raise NotImplmentedError


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # rm = ReplayMemory(capacity=env_config["memo"])
    # sample = torch.tensor(memory.sample(dqn.batch_size)).to(device)
    sample = memory.sample(dqn.batch_size)

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    obs = torch.cat(sample[0], dim=0).to(device)
    act = torch.cat(sample[1], dim=0).to(device)
    q_all = dqn.forward(obs).to(device)
    q_values = torch.gather(q_all, dim=1, index=act.long()).to(device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #     #       corresponding to the chosen actions.

    next_obs = sample[2]
    reward = sample[3]
    shape = (len(next_obs), 1)
    q_value_targets = torch.ones(shape).to(device)
    for i in range(len(next_obs)):
        if torch.is_tensor(next_obs[i]) is False:
            q_value_targets[i] = reward[i]
        else:
            q = target_dqn.forward(next_obs[i]).to(device)
            max_q_target = torch.max(q).to(device)
            #q_value_targets[i] = reward[i]+target_dqn.gamma*max_q_target
            q_value_targets[i] =  torch.tensor(reward[i]).to(device)+target_dqn.gamma*max_q_target
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    
    # Compute loss.
    # loss = F.mse_loss(q_values.squeeze(), q_value_targets)
    loss = F.mse_loss(q_values.squeeze(), q_value_targets.squeeze())

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
