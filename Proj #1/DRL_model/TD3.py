import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Replay_buffer:
    def __init__(self, max_size=50000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):

        index = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in index:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class ConvNet(nn.Module):
    def __init__(self, frame_size):
        super(ConvNet, self).__init__()

        self.frame_size = frame_size
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding='valid')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding='valid')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='valid')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=7, stride=1, padding='valid')

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 256)
        self.initialize_weights()

    def forward(self, x):
        x = x.reshape((-1, 4, self.frame_size, self.frame_size))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # [-1, 512]

        x = self.flatten(x)

        x = self.fc1(x)

        return x

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, s_dim, model):
        super(Actor, self).__init__()
        self.conv = model

        self.fc1 = nn.Linear(s_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.action_r = nn.Linear(100, 1)
        self.action_theta = nn.Linear(100, 1)
        self.initialize_weights()

    def forward(self, x):
        x = self.conv(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_r = torch.sigmoid(self.action_r(x))
        action_theta = torch.tanh(self.action_theta(x))

        action = torch.cat([action_r, action_theta], dim=1)

        return action

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# Critic
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, model):
        super(Critic, self).__init__()
        self.conv = model

        self.fc1 = nn.Linear(s_dim + a_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

        self.fc4 = nn.Linear(s_dim + a_dim, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 1)
        self.initialize_weights()

    def forward(self, x, u):
        x = self.conv(x)

        x1 = F.relu(self.fc1(torch.cat([x, u], dim=1)))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(torch.cat([x, u], dim=1)))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)

        return x1, x2

    def Q1(self, x, u):
        x = self.conv(x)

        x1 = F.relu(self.fc1(torch.cat([x, u], dim=1)))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        return x1

    def initialize_weights(self):
        # track all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class TD3(object):

    def __init__(self, state_dim, action_dim, action_lr, critic_lr, model1, model2):
        self.actor = Actor(state_dim, model1).to(device)
        self.actor_target = Actor(state_dim, model1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=action_lr)

        self.critic = Critic(state_dim, action_dim, model2).to(device)
        self.critic_target = Critic(state_dim, action_dim, model2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_it = 0
        self.replay_buffer = Replay_buffer()

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(device)
        action_r, action_theta = self.actor(state).cpu().data.numpy().flatten()

        if noise != 0:
            action_r = (action_r + (np.random.normal(0, noise, size=1))).clip(0, 1)
            action_theta = (action_theta + np.random.normal(0, noise * 2, size=1)).clip(-1, 1)

        action_r = torch.from_numpy(action_r).reshape(-1, 1)
        action_theta = torch.from_numpy(action_theta).reshape(-1, 1)

        action = torch.cat([action_r, action_theta], dim=1)

        return action.cpu().data.numpy().flatten()

    def update(self, batch_size, iterations, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            x, y, u, r, d = self.replay_buffer.sample(batch_size=batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            with torch.no_grad():
                next_action = self.actor_target(next_state)

                r_noise = (torch.randn(size=(batch_size,)) * policy_noise / 2).clamp(-noise_clip / 4,
                                                                                     noise_clip / 4).to(device)
                theta_noise = (torch.randn(size=(batch_size,)) * policy_noise).clamp(-noise_clip / 2,
                                                                                     noise_clip / 2).to(device)

                next_action[:, 0] += r_noise
                next_action[:, 1] += theta_noise

                next_action[:, 0] = next_action[:, 0].clamp(0, 1)
                next_action[:, 1] = next_action[:, 1].clamp(-1, 1)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:

                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory, epoch):
        torch.save(self.actor, directory + '/actor' + '/actor_{}.pt'.format(epoch))
        torch.save(self.critic, directory + '/critic' + '/critic_{}.pt'.format(epoch))
        print('')
        print('=' * 50)
        print('Epoch : {} // Model saved...'.format(epoch))
        print('=' * 50)

    def load(self, directory, epoch, device):
        self.actor = torch.load(directory + '/actor' + '/actor_{}.pt'.format(epoch), map_location=torch.device(device))
        self.critic = torch.load(directory + '/critic' + '/critic_{}.pt'.format(epoch),
                                 map_location=torch.device(device))
        print('')
        print('=' * 50)
        print('Model has been loaded...')
        print('=' * 50)
        self.actor.eval()
        self.critic.eval()


def make_model():
    frame_size = 84
    state_dim = 256
    action_dim = 2
    action_lr = 1e-4
    critic_lr = 1e-4
    model1 = ConvNet(frame_size)
    model2 = ConvNet(frame_size)
    agent = TD3(state_dim, action_dim, action_lr, critic_lr, model1, model2)

    return agent
