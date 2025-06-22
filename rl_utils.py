from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                
                # 安全的reset处理 - 避免解包错误
                obs_and_info = env.reset()
                if hasattr(obs_and_info, '__len__') and len(obs_and_info) > 1 and not isinstance(obs_and_info, np.ndarray):
                    # 新版本Gym返回(observation, info)
                    state = obs_and_info[0]
                else:
                    # 旧版本或numpy数组
                    state = obs_and_info
                
                done = False
                step_count = 0
                max_steps = 1000  # 防止无限循环
                
                while not done and step_count < max_steps:
                    action = agent.take_action(state)
                    
                    # 安全的step处理
                    step_return = env.step(action)
                    
                    if len(step_return) == 5:
                        # 新版本Gym: (observation, reward, terminated, truncated, info)
                        next_state, reward, terminated, truncated, info = step_return
                        done = terminated or truncated
                    elif len(step_return) == 4:
                        # 旧版本Gym: (observation, reward, done, info)
                        next_state, reward, done, info = step_return
                    else:
                        print(f"Warning: Unexpected step return length: {len(step_return)}")
                        break
                    
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    
                    state = next_state
                    episode_return += reward
                    step_count += 1
                    
                return_list.append(episode_return)
                
                # 只有当有有效的转换数据时才更新
                if len(transition_dict['states']) > 0:
                    agent.update(transition_dict)
                
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                
                # 安全的reset处理 - 避免解包错误
                obs_and_info = env.reset()
                if hasattr(obs_and_info, '__len__') and len(obs_and_info) > 1 and not isinstance(obs_and_info, np.ndarray):
                    # 新版本Gym返回(observation, info)
                    state = obs_and_info[0]
                else:
                    # 旧版本或numpy数组
                    state = obs_and_info
                
                done = False
                step_count = 0
                max_steps = 1000  # 防止无限循环
                
                while not done and step_count < max_steps:
                    action = agent.take_action(state)
                    
                    # 安全的step处理
                    step_return = env.step(action)
                    
                    if len(step_return) == 5:
                        # 新版本Gym: (observation, reward, terminated, truncated, info)
                        next_state, reward, terminated, truncated, info = step_return
                        done = terminated or truncated
                    elif len(step_return) == 4:
                        # 旧版本Gym: (observation, reward, done, info)
                        next_state, reward, done, info = step_return
                    else:
                        print(f"Warning: Unexpected step return length: {len(step_return)}")
                        break
                    
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    step_count += 1
                    
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                        
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                