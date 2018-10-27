import torch
import torch.nn as nn
import copy 

class PolicyApprox(nn.Module):
    def __init__(self,dim_in,hidden,dim_out):
        super(PolicyApprox, self).__init__()
        self.fc1 = nn.Linear(dim_in,hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden,dim_out)
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
  
class PolicyValueApprox(nn.Module):
    def __init__(self,dim_in,hidden,dim_out):
        super(PolicyValueApprox, self).__init__()
        self.fc1 = nn.Linear(dim_in,hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden,dim_out)
        self.softmax = nn.Softmax(dim=0)
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        probs = self.softmax(x[:-1])
        value = self.softplus(x[[-1]])
        return probs, value

class ReinforceLoss(nn.Module):
    '''
    Vanilla REINFORCE loss without baseline
    '''
    def __init__(self):
        super(ReinforceLoss, self).__init__()

    def forward(self,probs,rewards,gamma):
        ''' 
        Input: 
        - probs: list of probabilities associated with chosen actions in a trajectory
        - rewards: list of rewards in a trajectory
        - gamma: discount
        '''
        rewards_t = self._get_timestep_rewards(rewards,gamma)
        loss = 0
        for p, r in zip(probs,rewards_t):
            loss = loss + torch.log(p)*r
        return -loss #gradient ascent

    def _get_timestep_rewards(self,rewards,gamma):
        rewards_t = []
        rewards_accumulated = 0
        for r in reversed(rewards):
            rewards_t.insert(0,gamma*rewards_accumulated + r)
            rewards_accumulated += r
        return rewards_t

class PPOLoss(nn.Module):
    '''
    Proximal Policy Optimization with Generalized Advantage Estimators
    Sources:
    - High-dimensional continuous control using Generalized Advantage Estimation, Schulman et al, 2016
    - Proximal Policy Optimization Algorithms, Schulman et al, 2017
    '''
    def __init__(self, eps, gamma, lamb, c1, c2):
        ''' 
        Initialize loss function
        Input: 
        - eps: clipping hyperparameter
        - gamma: discount used for GAE 
        - lambda: exponential weight used for GAE
        - c1: coefficient for value function loss
        - c2: coefficient for entropy loss
        '''
        super(PPOLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.lamb = lamb
        self.entropy_loss = EntropyLoss()
        self.value_loss = nn.MSELoss(size_average=False,reduce=True)
        self.c1 = c1
        self.c2 = c2

    def forward(self, probs, probs_old, rewards, values):
        '''
        Input:
        - probs: T-dimensional tensor with probability of chosen action in the trajectory
        - probs_old: T-dimensional tensor with probability of chosen action from previous iteration
        - rewards: T-dimensional tensor with rewards of trajectory
        - values: T-dimensional tensor with value of every state visited in the trajectory 
        '''
        T = probs.size(0)
        advantages = self._calc_gaes(rewards, values)
        # calculate clipped loss
        ratio = probs/probs_old
        clip_loss = -torch.sum(torch.min(ratio*advantages, torch.clamp(ratio,1-self.eps,1+self.eps)*advantages))
        # add value function loss and entropy loss
        values_target = self._calc_value_targets(rewards)
        loss = clip_loss + self.c1*self.value_loss(values,values_target) + self.c2*self.entropy_loss(probs)
        return loss

    def _calc_gaes(self, rewards, values):
        '''
        Input:
        - rewards: T-dimensional tensor with rewards of trajectory
        - values: T-dimensional tensor with value of every state visited in the trajectory 
        '''
        T = rewards.size(0)
        gaes = torch.zeros(T)
        gae = 0
        # calculate GAE for every timestep t
        for t in reversed(range(T)):
            # calculate delta_{t},...,delta_{T-2},delta_{T-1}
            if t == T-1:
                delta = -values[t]+rewards[t]            
            else:
                delta = -values[t]+rewards[t]+self.gamma*values[t+1]
            # calculate GAE
            gae = (self.gamma*self.lamb)*gae + delta 
            gaes[t] = gae
        return gaes

    def _calc_value_targets(self, rewards):
        '''
        Input:
        - rewards: T-dimensional tensor with rewards of trajectory
        '''
        T = rewards.size(0)
        values_target = torch.zeros(T)
        reward_accumulated = 0
        for t in reversed(range(T)):
            reward_accumulated = reward_accumulated*self.gamma + rewards[t]
            values_target[t] = reward_accumulated
        return values_target


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, probs):
        '''
        Input:
        - probs: T-dimensional tensor with probability of chosen action in the trajectory
        '''
        entropy = probs*torch.log(probs+1e-5)
        loss = -torch.sum(entropy)
        return loss

class PolicyGradient():
    '''
    General class for policy gradient algorithms
    '''
    def __init__(self, policy_net, loss, optimizer, config):
        '''
        Input:
        - policy_net: neural used net as policy approximator
        - loss: algorithm-dependent loss function
        - optimizer
        - config: dictionary with 
        -- max_episodes
        -- max_timesteps
        -- batch_size: number of trajectories used for update
        -- gamma: discount of rewards
        '''
        self.max_e = config.get("max_episodes",1000)
        self.max_t = config.get("max_timesteps",100)
        self.batch_size = config.get("batch_size",1) 
        self.gamma = config.get("gamma",1)
        self.policy = policy_net
        self.optimizer = optimizer
        self.loss = loss

    def train(self, env, render=False):
        '''
        Input: 
        - env: OpenAI gym environment
        - render: displays training in environment
        '''
        self.optimizer.zero_grad()
        episode_length = []
        losses = []
        probs = []
        rewards = []
        loss = 0

        for ep in range(self.max_e):
            probs.clear()
            rewards.clear()
            observation = env.reset()
            done = False
            for t in range(self.max_t):
                if render == True:
                    env.render()
                policy_probs = self.policy(torch.tensor(data=observation, dtype=torch.float32))
                action = torch.multinomial(input=policy_probs, num_samples=1)[0].numpy()
                probs.append(policy_probs[action]) #add to memory
                observation, reward, done, _ = env.step(action)
                rewards.append(reward) #add to memory
                if done == True or t == self.max_t-1:
                    break
            loss += self.loss(probs,rewards,self.gamma)
            if ep % self.batch_size == 0:        
                # gradient ascent
                loss = loss/self.batch_size
                loss.backward()
                losses.append(loss.detach().numpy())
                self.optimizer.step()
                self.optimizer.zero_grad()               
                loss = 0
                episode_length.append(t)
        env.close()
        return losses, episode_length


class ProximalPolicyGradient():
    def __init__(self, policy_val_net, loss, optimizer, config):
        self.max_e = config.get("max_episodes",1000)
        self.max_t = config.get("max_timesteps",100)
        self.batch_size = config.get("batch_size",1) 

        self.policy = policy_val_net
        #self.policy_old = policy_val_net_old
        self.policy_old = copy.deepcopy(policy_val_net)
        
        self.loss = loss
        self.optimizer = optimizer

    def train(self, env, render=False):
        rewards = []
        probs = []
        probs_old = []
        values = []
        loss = 0

        episode_length = []
        losses = []

        for ep in range(self.max_e):
            rewards.clear()
            probs.clear()
            probs_old.clear()
            values.clear()
            observation = env.reset()
            #self.policy_old.load_state_dict(self.policy.state_dict())

            for t in range(self.max_t):
                if render == True:
                    env.render()
                probs_pol, value = self.policy(torch.tensor(data=observation,dtype=torch.float32))
                probs_pol_old, _ = self.policy_old(torch.tensor(data=observation,dtype=torch.float32))
                action = torch.multinomial(input=probs_pol_old, num_samples=1)[0].numpy()
                observation, reward, done, _ = env.step(action)
                # add to memory
                rewards.append(torch.unsqueeze(torch.tensor(reward),0))
                probs.append(torch.unsqueeze(probs_pol[[action]],0))
                probs_old.append(torch.unsqueeze(probs_pol_old[action],0))
                values.append(value)
                if done == True or t == self.max_t-1:
                    break
            # calculate loss
            probs_tensor = torch.cat(probs)
            probs_old_tensor = torch.cat(probs_old)
            rewards_tensor = torch.cat(rewards)
            values_tensor = torch.cat(values)
            loss += self.loss(probs=probs_tensor, probs_old=probs_old_tensor, rewards=rewards_tensor, values=values_tensor)
            if ep % self.batch_size == 0:
                # gradient ascent
                #self.policy_old.load_state_dict(self.policy.state_dict())
                self.policy_old = copy.deepcopy(self.policy)
                loss = loss/self.batch_size
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()               
                losses.append(loss.detach().numpy())
                loss = 0
                episode_length.append(t)
        env.close()
        return losses, episode_length