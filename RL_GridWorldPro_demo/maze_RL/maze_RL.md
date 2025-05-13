# Q-Learning Maze Discussion

## 1. Why is the Q-table "State × Action"?

### ✅ Q-table Definition
The Q-table stores the estimated value (expected total reward) for each **state-action pair**:

```python
Q[state, action] → expected total reward for taking an action in a given state
```

### ✅ Why State × Action?
- **State**: The current situation or position of the agent (e.g., cell index 0–15 in a 4x4 grid).
- **Action**: Possible moves the agent can take (e.g., up, down, left, right).

To choose the optimal action for each state, we need to store and update the value of every possible **state-action combination**.

---

## 2. Epsilon-Greedy Action Selection

### ✅ Code Snippet:

```python
# Epsilon-greedy action selection
if np.random.uniform(0, 1) < EPSILON:
    action = np.random.choice(ACTIONS)  # Explore: choose random action
else:
    action = ACTIONS[np.argmax(Q[state])]  # Exploit: choose best known action
```

### ✅ What is Epsilon-Greedy?

The **Epsilon-greedy** strategy is used to balance **exploration** and **exploitation** during training:

| Term        | Meaning                                                                 |
|-------------|-------------------------------------------------------------------------|
| **Exploration** | Try random actions to discover new, possibly better strategies.        |
| **Exploitation** | Choose the best-known action based on current Q-values (greedy step). |

### ✅ How It Works:

- With probability **ε (epsilon)** (e.g. 0.1), the agent picks a **random action** (explore).
- With probability **1 - ε**, the agent chooses the **best-known action** from the Q-table (exploit).

This helps the agent avoid getting stuck in local optima and ensures it learns a more optimal policy over time.

---

## 3. Q-table Update Formula and Its Physical Meaning

### ✅ Q-Learning Update Formula:

```python
Q[state, action] += α * (reward + γ * max(Q[next_state]) - Q[state, action])
```

### ✅ Parameters Explained:

| Parameter       | Symbol       | Meaning                                                   |
|----------------|--------------|------------------------------------------------------------|
| Learning rate  | α (ALPHA)    | How much new information overrides the old one (0–1).      |
| Discount factor| γ (GAMMA)    | Importance of future rewards (closer to 1 = long-term focus). |
| Immediate reward | r          | The direct reward received after taking the action.        |
| Max future value | max(Q[next_state]) | The best possible Q-value in the next state.       |

### ✅ Intuition Behind the Formula:

1. **Current estimate**: `Q[state, action]`
2. **New target**: `reward + γ * max(Q[next_state])`
3. **TD Error**: Difference between the new target and current estimate.
4. **Update**: Gradually adjust the current estimate towards the target using `α`.

This is the core idea behind **Temporal Difference Learning (TD Learning)**.

---

## 4. Example Calculation

Suppose the agent is in state `0` and takes action `'right'` to reach state `1`:

- Current Q-value: `Q[0, right] = 0`
- Reward: `-1`
- Max Q-value in next state: `max(Q[1]) = 0`

Update:

```python
Q[0, right] += 0.1 * (-1 + 0.99 * 0 - 0)
             = -0.1
```

This means moving right from state 0 currently has a negative expected value.

---

## 5. Summary

| Question                           | Answer |
|------------------------------------|--------|
| Why is the Q-table "State × Action"? | To store and choose the best action for every possible state. |
| What is the basis of the Q update formula? | Based on the **Bellman Optimality Equation** and **TD learning**, combining immediate and future rewards. |
| What is Epsilon-greedy action selection? | A strategy to balance exploration (random actions) and exploitation (greedy actions) using a probability ε. |

