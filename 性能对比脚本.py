import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 原始版本的REINFORCE
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class OriginalREINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        reward_list = transition_dict['rewards']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

# 优化版本的REINFORCE
class OptimizedREINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device
        
        # 启用cuDNN优化
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        # GPU预热
        self._warmup_gpu()
        
        # 预分配张量
        self._preallocate_tensors()

    def _warmup_gpu(self):
        if self.device.type == 'cuda':
            dummy_input = torch.randn(32, 4, device=self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.policy_net(dummy_input)
            torch.cuda.synchronize()
    
    def _preallocate_tensors(self):
        self._temp_state = torch.zeros(1, 4, device=self.device)

    def take_action(self, state):
        self._temp_state[0] = torch.from_numpy(state).float()
        with torch.no_grad():
            probs = self.policy_net(self._temp_state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

    def update(self, transition_dict):
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']

        # 批量转换到GPU
        states_tensor = torch.tensor(np.array(states), dtype=torch.float, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        # 在GPU上计算returns
        returns = torch.zeros(len(rewards), device=self.device)
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G
        
        # 标准化
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 批量计算
        probs = self.policy_net(states_tensor)
        log_probs = torch.log(probs.gather(1, actions_tensor.unsqueeze(1))).squeeze()
        policy_loss = -(log_probs * returns).mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

def train_agent(agent_class, name, device, num_episodes=200):
    """训练单个agent并返回性能统计"""
    print(f"\n=== 训练 {name} ===")
    
    # 环境设置
    env = gym.make("CartPole-v1", render_mode=None)
    env.reset(seed=42)  # 使用相同的种子确保公平比较
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = agent_class(state_dim, 128, action_dim, 1e-3, 0.98, device)
    
    return_list = []
    training_times = []
    
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc=f"训练{name}"):
        episode_start = time.time()
        
        episode_return = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
        done = False
        state, _ = env.reset()
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated
            
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            
            state = next_state
            episode_return += reward
        
        return_list.append(episode_return)
        agent.update(transition_dict)
        
        episode_time = time.time() - episode_start
        training_times.append(episode_time)
    
    total_time = time.time() - start_time
    env.close()
    
    return {
        'name': name,
        'total_time': total_time,
        'avg_episode_time': np.mean(training_times),
        'return_list': return_list,
        'final_avg_return': np.mean(return_list[-50:]),
        'max_return': max(return_list),
        'training_times': training_times
    }

def compare_performance():
    """比较两个版本的性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 训练两个版本
    original_results = train_agent(OriginalREINFORCE, "原始版REINFORCE", device)
    optimized_results = train_agent(OptimizedREINFORCE, "优化版REINFORCE", device)
    
    # 打印对比结果
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    
    print(f"\n训练时间对比:")
    print(f"原始版总时间: {original_results['total_time']:.2f}秒")
    print(f"优化版总时间: {optimized_results['total_time']:.2f}秒")
    speed_up = original_results['total_time'] / optimized_results['total_time']
    print(f"速度提升: {speed_up:.2f}x")
    
    print(f"\n每Episode平均时间:")
    print(f"原始版: {original_results['avg_episode_time']*1000:.1f}ms")
    print(f"优化版: {optimized_results['avg_episode_time']*1000:.1f}ms")
    episode_speedup = original_results['avg_episode_time'] / optimized_results['avg_episode_time']
    print(f"Episode速度提升: {episode_speedup:.2f}x")
    
    print(f"\n训练效果对比:")
    print(f"原始版最终平均回报: {original_results['final_avg_return']:.2f}")
    print(f"优化版最终平均回报: {optimized_results['final_avg_return']:.2f}")
    print(f"原始版最高回报: {original_results['max_return']:.1f}")
    print(f"优化版最高回报: {optimized_results['max_return']:.1f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 回报对比
    plt.subplot(2, 3, 1)
    plt.plot(original_results['return_list'], label='原始版', alpha=0.7)
    plt.plot(optimized_results['return_list'], label='优化版', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('训练回报对比')
    plt.legend()
    
    # 训练时间对比
    plt.subplot(2, 3, 2)
    plt.plot(original_results['training_times'], label='原始版', alpha=0.7)
    plt.plot(optimized_results['training_times'], label='优化版', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Time per Episode (s)')
    plt.title('每Episode训练时间对比')
    plt.legend()
    
    # 速度提升图
    plt.subplot(2, 3, 3)
    speedups = [original_results['training_times'][i] / optimized_results['training_times'][i] 
                for i in range(len(original_results['training_times']))]
    plt.plot(speedups)
    plt.xlabel('Episodes')
    plt.ylabel('Speed Up (x)')
    plt.title('实时速度提升')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # 累积时间对比
    plt.subplot(2, 3, 4)
    original_cumtime = np.cumsum(original_results['training_times'])
    optimized_cumtime = np.cumsum(optimized_results['training_times'])
    plt.plot(original_cumtime, label='原始版')
    plt.plot(optimized_cumtime, label='优化版')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Time (s)')
    plt.title('累积训练时间')
    plt.legend()
    
    # 性能统计柱状图
    plt.subplot(2, 3, 5)
    categories = ['总时间(s)', '平均Episode时间(ms)', '最终回报', '最高回报']
    original_values = [original_results['total_time'], 
                      original_results['avg_episode_time']*1000,
                      original_results['final_avg_return'],
                      original_results['max_return']]
    optimized_values = [optimized_results['total_time'],
                       optimized_results['avg_episode_time']*1000,
                       optimized_results['final_avg_return'],
                       optimized_results['max_return']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # 标准化数值以便显示
    orig_norm = [original_values[i]/max(original_values[i], optimized_values[i]) for i in range(len(categories))]
    opt_norm = [optimized_values[i]/max(original_values[i], optimized_values[i]) for i in range(len(categories))]
    
    plt.bar(x - width/2, orig_norm, width, label='原始版', alpha=0.7)
    plt.bar(x + width/2, opt_norm, width, label='优化版', alpha=0.7)
    plt.xlabel('指标')
    plt.ylabel('标准化值')
    plt.title('性能对比（标准化）')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    
    # 回报分布对比
    plt.subplot(2, 3, 6)
    plt.hist(original_results['return_list'][-50:], bins=15, alpha=0.5, label='原始版', density=True)
    plt.hist(optimized_results['return_list'][-50:], bins=15, alpha=0.5, label='优化版', density=True)
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('最后50个Episode回报分布')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # GPU内存使用情况
    if device.type == 'cuda':
        print(f"\nGPU内存使用:")
        print(f"已分配: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"缓存: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    
    return original_results, optimized_results
    
if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    original_results, optimized_results = compare_performance() 