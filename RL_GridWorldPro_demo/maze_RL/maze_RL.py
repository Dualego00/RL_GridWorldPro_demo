import numpy as np

# 环境参数
GRID_SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
NUM_ACTIONS = 4
NUM_STATES = GRID_SIZE * GRID_SIZE

# 超参数
ALPHA = 0.1  # 学习率
GAMMA = 0.99  # 折扣因子
EPSILON = 0.1  # 探索率
EPISODES = 1000  # 训练轮数

# 初始化Q表 (状态 x 动作)
Q = np.zeros((NUM_STATES, NUM_ACTIONS))

# 定义奖励函数
def get_reward(state):
    if state == 15:  # 宝藏位置（4x4网格的右下角）
        return 10
    elif state == 5:  # 陷阱位置
        return -10
    else:
        return -1

# 状态转移函数
def move(state, action):
    x, y = state // GRID_SIZE, state % GRID_SIZE
    if action == 'up' and x > 0:
        x -= 1
    elif action == 'down' and x < GRID_SIZE-1:
        x += 1
    elif action == 'left' and y > 0:
        y -= 1
    elif action == 'right' and y < GRID_SIZE-1:
        y += 1
    return x * GRID_SIZE + y

# Q-learning 训练过程
for episode in range(EPISODES):
    state = 0  # 初始状态
    done = False
    
    while not done:
        # Epsilon-greedy 选择动作
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = ACTIONS[np.argmax(Q[state])]
        
        # 执行动作，获得下一个状态和奖励
        next_state = move(state, action)
        reward = get_reward(next_state)
        done = (next_state == 15)  # 终止条件
        
        # 更新Q表
        Q[state, ACTIONS.index(action)] += ALPHA * (
            reward + GAMMA * np.max(Q[next_state]) - Q[state, ACTIONS.index(action)]
        )
        
        state = next_state

# 输出训练后的Q表
print("Trained Q-table:")
print(Q)