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

class Point:
    """ A point object with an x and y coordinate """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class RaceCar:
    """ The race car environment. """

    def __init__(self, L1, L2, W1, W2, no_action_prob):
        """ Initializes the enviornment.

            The race track is as follows:


            -  ###########################################&
            |  ###########################################&
            |  ###########################################&
            W2 ###########################################&
            |  ###########################################&
            |  ###########################################&
            -  ###########################################&
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            W1 ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            |  ######################                     |
            -  $$$$$$$$$$$$$$$$$$$$$$                     |
               |--------L1----------|---------L2----------|

            The $ symbols is the starting line, and the & symbols is the finish line.

            @type L1: int
            @type L2: int
            @type W1: int
            @type W2: int
            @type no_action_prob: float
                A number between 0 and 1 representing the probability of an action
                failing to register.
        """

        self.L1 = L1
        self.L2 = L2
        self.W1 = W1
        self.W2 = W2
        self.no_action_prob = no_action_prob

        self.Vx = 0
        self.Vy = 0

        # Borders of the race track
        self.borders = []
        self.borders.append((Point(-1, -1), Point(L1 + 1, -1)))
        self.borders.append((Point(L1 + 1, -1), Point(L1 + 1, W1 - 1)))
        self.borders.append((Point(L1 + 1, W1 - 1), Point(L1 + L2, W1 - 1)))
        self.borders.append((Point(L1 + L2, W1 + W2 + 1), Point(-1, W1 + W2 + 1)))
        self.borders.append((Point(-1, W1 + W2 + 1), Point(-1, -1)))

        # End points of the finish line
        self.finish_p = Point(L1 + L2, W1 - 1)
        self.finish_q = Point(L1 + L2, W1 + W2 + 1)

    def reset(self):
        self.Vx, self.Vy = 0, 0
        return (random.randint(0, L1), 0), False

    def transitionState(self, state, action):
        """ The state transition function

            The action is an integer from 0 to 8, converted into one of
            nine different actions as described in the convertActionToVelocity
            function. The velocity is updated first, then the new position is
            calculated. If the trajectory intersects a track boundary, then
            the car is sent back to a random position on the starting line with
            the velocities set to 0. If the trajectory crosses the finish line, then
            the episode ends.

            @type state: tuple[int]
                A tuple (x,y) representing the current state.
            @type action: int
                An integer from 0 to 8 representing the action.
            @rtype: tuple
                A tuple ((x,y), done) representing the new x, y positions, and if the
                episode has ended (car crossed finish line).

        """
        x, y = state
        dV_x, dV_y = self.convertActionToVelocity(action)

        # Updating velocities with probability 1-no_action_prob
        # if the update does not result in both velocities equal to 0
        r = random.uniform(0, 1)
        if r > self.no_action_prob:
            if not (self.Vx + dV_x <= 0 and self.Vy + dV_y <= 0):
                self.Vx = max(0, min(5, self.Vx + dV_x))
                self.Vy = max(0, min(5, self.Vy + dV_y))

        new_x = int(x + self.Vx)
        new_y = int(y + self.Vy)

        # Check if car has crossed a boundary
        for p, q in self.borders:
            if self.doIntersect(Point(x, y), Point(new_x, new_y), p, q):
                self.Vx = 0
                self.Vy = 0
                return -1, (random.randint(0, L1), 0), False

        # Check if car has crossed finish line
        if self.doIntersect(Point(x, y), Point(new_x, new_y), self.finish_p, self.finish_q):
            self.Vx = 0
            self.Vy = 0
            return -1, (random.randint(0, L1), 0), True

        return -1, (new_x, new_y), False

    def getAllNextStates(self, state):
        data = []
        orig_Vx = self.Vx
        orig_Vy = self.Vy

        for action in range(9):
            reward, next_state, done = self.transitionState(state, action)
            data.append((action, next_state, done))
            self.Vx, self.Vy = orig_Vx, orig_Vy

        return data

    def convertActionToVelocity(self, action):
        """ Converts the action (an integer from 0 to 8)
            to the changes in velocity in x and y
            according to the following table

                          dV_x
                    | -1    0    1
                  ----------------
                 -1 |  0    1    2
                    |
              dVy 0 |  3    4    5
                    |
                  1 |  6    7    8

            @type action: int
                A number between 0 and 8.
            @type: tuple[int]
                The changes in x velocity and y velocity
                represented as (dV_x, dV_y)

        """
        if not 0 <= action <= 8:
            raise ValueError("That's not a valid action!")

        dV_x = action % 3 - 1
        dV_y = action // 3 - 1

        return dV_x, dV_y

    def doIntersect(self, p1, q1, p2, q2):
        """ Function that returns True if the line segment 'p1q1'
            and 'p2q2' intersect.

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

            @type p1: Point
                Point 1 of line segment 1.
            @type q1: Point
                Point 2 of line segment 1.
            @type p2: Point
                Point 1 of line segment 2.
            @type q2: Point
                Point 2 of line segment 2.
            @rtype: boolean
        """

        # Find the 4 orientations required for
        # the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True

        # Special Cases

        # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True

        # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True

        # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True

        # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True

        # If none of the cases
        return False

    def orientation(self, p, q, r):
        """ Finds the orientation of an ordered triplet (p,q,r)
            function returns the following values:
            0 : Colinear points
            1 : Clockwise points
            2 : Counterclockwise

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

            @type p: Point
            @type q: Point
            @type r: Point
            @rtype: int

        """

        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):

            # Clockwise orientation
            return 1
        elif (val < 0):

            # Counterclockwise orientation
            return 2
        else:

            # Colinear orientation
            return 0

    def onSegment(self, p, q, r):
        """ Given three colinear points p, q, r, the function checks if
            point q lies on line segment 'pr'

            This piece of code was taken from:
            https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

            @type p: Point
            @type q: Point
            @type r: Point
            @rtype: boolean

        """
        if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False


Transition = namedtuple('Transition', ('state', 'next_state', 'reward', 'done'))


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

    def __init__(self, env, NN, NN_target, optimizer, criterion, RBF):
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
        self.NN_target = NN_target
        self.optimizer = optimizer
        self.criterion = criterion
        self.RBF = RBF

    def chooseAction(self, epsilon, state):
        """ Chooses action. With probability epsilon it will choose
            an exploratory action. Otherwise, it will choose
            an action which maximizes the estimated reward of all
            possible next states.

            @type epsilon: float
                Exploration probability.
            @rtype: int
                An integer representing the action.
        """
        if random.uniform(0, 1) < epsilon:
            return random.randrange(9)

        cur_best_val = -float('inf')
        cur_best_action = 0

        data = env.getAllNextStates(state)

        with torch.no_grad():
            for action, next_state, done in data:
                if next_state != state:
                    value = self.NN(self.RBF[next_state]).item() if not done else 0
                    if value > cur_best_val:
                        cur_best_val = value
                        cur_best_action = action
        #print(data)
        return cur_best_action

    def optimizeModel(self, memory, batch_size, gamma):
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
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        # Predictions and targets
        predictions = self.NN(state_batch)
        with torch.no_grad():
            targets = reward_batch + gamma * self.NN(next_state_batch) * (~done_batch)

        # Loss and gradient descent
        loss = self.criterion(predictions, targets)

        x = loss.item()

        loss.backward()

        self.optimizer.step()

        self.optimizer.zero_grad()

        return x

    def train(self, episodes, epsilon_initial, epsilon_min, epsilon_stop_episode,
              network_update_freq, gamma, memory_capacity, batch_size):
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

        tot_steps = 0
        running_loss = 0

        depsilon = (epsilon_initial-epsilon_min)/epsilon_stop_episode

        for episode in range(episodes):

            if epsilon_initial > epsilon_min:
                epsilon_initial -= depsilon

            if episode % network_update_freq == 0:
                # Update target network
                self.NN_target.load_state_dict(self.NN.state_dict())

            if (episode + 1) % 10 == 0:
                print(f'Episode {episode + 1}/{episodes} completed!')
                print(f'Average steps per episode: {tot_steps / 10}')
                writer.add_scalar('training loss', running_loss / tot_steps, episode)
                self.plotValue()
                tot_steps = 0
                running_loss = 0

            state, done = self.env.reset()


            while not done:
                tot_steps += 1

                action = self.chooseAction(epsilon_initial, state)

                reward, next_state, done= self.env.transitionState(state, action)

                #score += reward
                reward = torch.tensor([[reward]], device=device)
                done = torch.tensor([[done]], device=device)

                # Saves the transition
                memory.push(self.RBF[state], self.RBF[next_state], reward, done)

                # Perform one step of batch gradient descent
                running_loss += self.optimizeModel(memory, batch_size, gamma)

                state = next_state

        writer.close()

    def plotValue(self):
        """
        Plots and saves the current policy in x and y.
        """
        V = np.zeros((self.env.W1 + self.env.W2 + 1, self.env.L1 + self.env.L2 + 1))

        for y in range(len(V)):
            for x in range(len(V)):
                if ((0 <= x <= L1 and 0 <= y <= W1 + W2) or (L1 <= x <= L1 + L2 and W1 <= y <= W1 + W2)):
                    with torch.no_grad():
                        value = self.NN(self.RBF[(x,y)]).item()
                    V[y][x] = value

        # Plotting and saving results
        plt.figure()
        plt.imshow(V, origin='lower', interpolation='none')
        plt.title('Value')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig('Value_racecar.png')
        plt.close()

def constructRBFStates(L1, L2, W1, W2, sigma):
    """
    Constructs a dictionary dict[tuple] -> torch.tensor that converts
    tuples (x,y) representing positions to torch tensors used as input to the
    neural network. The tensors have an entry for each valid position on the
    race track. For each position (x,y), the tensor is constructed using the gaussian
    radial basis function with standard deviation sigma. In other words, if entry i corresponds
    to the position p2 = (x2, y2), then the tensor for a point p1 = (x1,y1) will have
    tensor[i] = Gaussian_RBF(p1, p2).

    @type L1: int
        See description in the @RaceCar class.
    @type L2: int
        See description in the @RaceCar class.
    @type W1: int
        See description in the @RaceCar class.
    @type W2: int
        See description in the @RaceCar class.
    @type sigma: float
        The standard deviation of the gaussian radial basis function.
    """
    N_states = (L1+1)*(W1+W2+1)+L2*(W2+1)
    x_coords = torch.zeros(N_states, dtype=torch.float32)
    y_coords = torch.zeros(N_states, dtype=torch.float32)
    state_to_basis = {}
    ind = 0
    for x in range(L1+L2+1):
        for y in range(W1+W2+1):
            if (0<=x<=L1 and 0<=y<=W1+W2) or (0<=x<=L1+L2 and W1<=y<=W1+W2):
                x_coords[ind] = x
                y_coords[ind] = y
                ind += 1

    for x in range(L1 + L2 + 1):
        for y in range(W1 + W2 + 1):
            if (0 <= x <= L1 and 0 <= y <= W1 + W2) or (0 <= x <= L1 + L2 and W1 <= y <= W1 + W2):
                basis = torch.exp(-((x_coords-x)**2 + (y_coords-y)**2)/(2*sigma**2))
                state_to_basis[(x,y)] = basis.view(1, -1).to(device)

    return state_to_basis

if __name__ == "__main__":
    # Track parameters
    L1 = 10
    L2 = 10
    W1 = 20
    W2 = 5
    no_action_prob = 0.1

    # Gaussian RBF standard deviation
    sigma = 1

    # Training parameters
    episodes = 1000000
    gamma = 1.0
    learning_rate = 1e-3
    epsilon_initial = 0.05
    epsilon_min = 0.05
    epsilon_stop_episode = 500
    memory_capacity = 2000
    batch_size = 256
    network_update_freq = 100

    # Network parameters
    input_size = (L1+1)*(W1+W2+1)+L2*(W2+1)

    state_to_basis = constructRBFStates(L1, L2, W1, W2, sigma)

    env = RaceCar(L1, L2, W1, W2, no_action_prob)
    model_value = nn.Linear((L1+1)*(W1+W2+1)+L2*(W2+1), 1, bias=False).to(device)
    model_value.weight.data.fill_(0)

    #model_value.load_state_dict(torch.load('race_car_NN_value_model'))
    model_value_target = copy.deepcopy(model_value)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model_value.parameters(), lr=learning_rate)

    tetris_agent = Agent(env, model_value, model_value_target, optimizer, criterion, state_to_basis)

    tetris_agent.train(episodes, epsilon_initial, epsilon_min, epsilon_stop_episode,
              network_update_freq, gamma, memory_capacity, batch_size)