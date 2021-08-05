import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque

writer = SummaryWriter("runs/state_value_network")


#% matplotlib notebook

# GPU if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataGen:
    def transitionState(self):
        x = random.uniform(0, 3)
        return torch.tensor([[x]], dtype=torch.float32).to(device), torch.tensor([[np.sin(2*x)-np.cos(x)]], dtype=torch.float32).to(device), torch.tensor([[random.uniform(0, 1)<0.05]]).to(device)

Transition = namedtuple('Transition', ('state', 'value', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        """
        Initializes the replay memory.
        @type capacity: int
            The capacity of the memory.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition.
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Samples a batch of transitions of size batch_size
        randomly.
        @type batch_size: int
            The size of the batch.
        @rtype: list[tuple]
            A list of the sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    """ Agent object that uses the actor-critic network to find the
        optimal policy.
    """

    def __init__(self, env, NN, optimizer, criterion):
        """ Initializes the agent.

            @type env: Tetris
                The Tetris environment.
            @type NN: NeuralNet
                Neural network for computing the state-action values.
            @type NN_target: NeuralNet
                Target neural network for estimating state-action values.
            @type optimizer: torch.optim
                Torch optimizer object.
            @type criterion: nn loss
                Neural network loss function.
            """
        self.env = env
        self.NN = NN
        self.optimizer = optimizer
        self.criterion = criterion

    def optimizeModel(self, memory, batch_size):
        """
        Performs one step of mini-batch gradient descent.

        @type memory: ReplayMemory
            The replay memory from which to draw the experience.
        @type batch_size: int
            The mini-batch size.
        @type gamma: float
            The discount factor.
        """
        batch_size = min(len(memory), batch_size)

        # Sampling experiences
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Unpacking data
        state_batch = torch.cat(batch.state)
        value_batch = torch.cat(batch.value)
        done_batch = torch.cat(batch.done)

        # Predictions and targets
        predictions = self.NN(state_batch)
        with torch.no_grad():
            value_batch *= ~done_batch

        self.optimizer.zero_grad()

        # Loss and gradient descent
        loss = self.criterion(predictions, value_batch)

        x = loss.item()

        loss.backward()

        self.optimizer.step()


        return x

    def train(self, episodes, memory_capacity, batch_size):
        """ Trains the agent using the actor-critic method with eligibility traces.

            @type episodes: int
                The number of episodes to train.
            @type epsilon: float
                The exploration probability.
            @type network_update_freq: int
                Number of episodes before copying the network
                parameters to the delayed network.
            @type gamma: float
                The discount factor.
            @type memory_capacity: int
                The capacity of the replay memory.
            @type batch_size: int
                Mini-batch size for training.

        """

        memory = ReplayMemory(memory_capacity)

        running_loss = 0

        tot_steps = 0

        for episode in range(episodes):

            if (episode + 1) % 100 == 0:
                print(f'Episode {episode + 1}/{episodes} completed!')
                print(f'Average steps per episode: {tot_steps / 100}')
                print(f'Average loss per episode: {running_loss / 100}')
                writer.add_scalar('training loss', running_loss / 100.0, episode)
                tot_steps = 0
                running_loss = 0

            done = False

            while not done:
                tot_steps += 1

                state, value, done = self.env.transitionState()

                # Saves the transition
                memory.push(state, value, done)

                # Perform one step of batch gradient descent
                running_loss += self.optimizeModel(memory, batch_size)


        writer.close()


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


if __name__ == "__main__":
    # Network parameters
    input_size = 1
    hidden_size1 = 10
    hidden_size2 = 10

    learning_rate = 1e-3
    episodes = 400
    memory_capacity = 1000
    batch_size = 512

    env = DataGen()
    model_value = QNetwork(input_size, hidden_size1, hidden_size2).to(device)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model_value.parameters(), lr=learning_rate)

    #model_value.load_state_dict(torch.load('tetris_NN_value_model'))

    tetris_agent = Agent(env, model_value, optimizer, criterion)

    tetris_agent.train(episodes, memory_capacity, batch_size)

    plt.figure()
    xvals = np.linspace(0, 3, 100)
    with torch.no_grad():
        yvals = [tetris_agent.NN(torch.tensor([[x]], dtype=torch.float32).to(device)).item() for x in xvals]
    plt.plot(xvals, yvals, label='estimated')
    plt.plot(xvals, np.sin(2*xvals)-np.cos(xvals), label='true')
    plt.legend()
    plt.show()