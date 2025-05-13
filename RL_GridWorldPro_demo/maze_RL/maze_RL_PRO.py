import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import List, Tuple

class MazeRL:
    def __init__(self):
        # Environment parameters
        self.GRID_SIZE = 4
        self.ACTIONS = ['up', 'down', 'left', 'right']
        self.NUM_ACTIONS = 4
        self.NUM_STATES = self.GRID_SIZE * self.GRID_SIZE
        
        # Hyperparameters
        self.ALPHA = 0.1  # Learning rate
        self.GAMMA = 0.99  # Discount factor
        self.EPSILON = 0.1  # Exploration rate
        self.EPISODES = 1000  # Training episodes
        
        # Initialize Q-table (state x action)
        self.Q = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
        
        # Initialize traps and treasure
        self.traps = []  # Will be populated during training
        self.treasure_pos = 15
        
        # Training time
        self.training_time = 0
        
        # Debug information
        self.episode_rewards = []
        self.episode_steps = []
        self.successful_episodes = 0
        self.trap_hits = 0
        
        self.setup_gui()
    
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Maze Reinforcement Learning")
        self.root.geometry("800x600")
        
        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="5")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Grid Size Input
        ttk.Label(control_frame, text="Grid Size:").grid(row=0, column=0, padx=5, pady=5)
        self.grid_size_var = tk.StringVar(value="4")
        grid_size_entry = ttk.Entry(control_frame, textvariable=self.grid_size_var, width=10)
        grid_size_entry.grid(row=0, column=1, padx=5, pady=5)
        grid_size_entry.bind('<KeyRelease>', self.update_max_traps)
        
        # Number of Traps Input
        ttk.Label(control_frame, text="Number of Traps:").grid(row=0, column=2, padx=5, pady=5)
        self.num_traps_var = tk.StringVar(value="1")
        self.traps_entry = ttk.Entry(control_frame, textvariable=self.num_traps_var, width=10)
        self.traps_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Max Traps Label
        self.max_traps_label = ttk.Label(control_frame, text="(Max: 14)")
        self.max_traps_label.grid(row=0, column=4, padx=5, pady=5)
        
        # Reward and Penalty Inputs
        ttk.Label(control_frame, text="Treasure Reward:").grid(row=1, column=0, padx=5, pady=5)
        self.treasure_reward_var = tk.StringVar(value="10")
        ttk.Entry(control_frame, textvariable=self.treasure_reward_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Trap Penalty:").grid(row=1, column=2, padx=5, pady=5)
        self.trap_penalty_var = tk.StringVar(value="-10")
        ttk.Entry(control_frame, textvariable=self.trap_penalty_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Learning Rate and Discount Factor
        ttk.Label(control_frame, text="Learning Rate (α):").grid(row=2, column=0, padx=5, pady=5)
        self.alpha_var = tk.StringVar(value="0.1")
        ttk.Entry(control_frame, textvariable=self.alpha_var, width=10).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Discount Factor (γ):").grid(row=2, column=2, padx=5, pady=5)
        self.gamma_var = tk.StringVar(value="0.99")
        ttk.Entry(control_frame, textvariable=self.gamma_var, width=10).grid(row=2, column=3, padx=5, pady=5)
        
        # Episodes and Max Steps Input
        ttk.Label(control_frame, text="Training Episodes:").grid(row=3, column=0, padx=5, pady=5)
        self.episodes_var = tk.StringVar(value="1000")
        ttk.Entry(control_frame, textvariable=self.episodes_var, width=10).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Max Steps:").grid(row=3, column=2, padx=5, pady=5)
        self.max_steps_var = tk.StringVar(value="8")
        ttk.Entry(control_frame, textvariable=self.max_steps_var, width=10).grid(row=3, column=3, padx=5, pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Initialize", command=self.initialize_environment).grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Start Training", command=self.start_training).grid(row=4, column=2, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Show Results", command=self.show_results).grid(row=5, column=0, columnspan=4, pady=10)
        
        # Status Label
        self.status_label = ttk.Label(control_frame, text="")
        self.status_label.grid(row=6, column=0, columnspan=4, pady=5)
        
        # Canvas for maze visualization
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg='white')
        self.canvas.pack(pady=10)
        
        # Initialize max traps label
        self.update_max_traps()
    
    def update_max_traps(self, event=None):
        """Update the maximum number of traps based on grid size"""
        try:
            grid_size = int(self.grid_size_var.get())
            max_traps = grid_size * grid_size - 2  # Total cells minus start and treasure
            self.max_traps_label.config(text=f"(Max: {max_traps})")
            
            # If current number of traps is greater than max, update it
            current_traps = int(self.num_traps_var.get())
            if current_traps > max_traps:
                self.num_traps_var.set(str(max_traps))
        except ValueError:
            pass
    
    def initialize_environment(self):
        try:
            # Update parameters from GUI
            self.GRID_SIZE = int(self.grid_size_var.get())
            self.NUM_STATES = self.GRID_SIZE * self.GRID_SIZE
            self.treasure_pos = self.NUM_STATES - 1  # Always in bottom-right corner
            num_traps = int(self.num_traps_var.get())
            
            # Validate number of traps
            max_traps = self.NUM_STATES - 2  # Total cells minus start and treasure
            if num_traps > max_traps:
                messagebox.showerror("Error", f"Number of traps must be less than or equal to {max_traps}")
                return
            
            self.treasure_reward = int(self.treasure_reward_var.get())
            self.trap_penalty = int(self.trap_penalty_var.get())
            
            # Reset Q-table
            self.Q = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
            
            # Generate random traps
            self.traps = []
            available_positions = list(range(self.NUM_STATES))
            available_positions.remove(self.treasure_pos)  # Remove treasure position from possible trap positions
            available_positions.remove(0)  # Remove start position from possible trap positions
            
            for _ in range(num_traps):
                if not available_positions:
                    break
                trap_pos = np.random.choice(available_positions)
                available_positions.remove(trap_pos)
                self.traps.append((trap_pos, self.trap_penalty))
            
            # Show initial state
            self.show_initial_state()
            
            self.status_label.config(text="Environment initialized successfully")
            messagebox.showinfo("Success", "Environment initialized successfully")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters")
    
    def show_initial_state(self):
        self.canvas.delete("all")
        
        # Calculate cell size
        cell_size = 400 / self.GRID_SIZE
        
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, 400, fill='black')
            self.canvas.create_line(0, i * cell_size, 400, i * cell_size, fill='black')
        
        # Draw traps
        for trap_pos, _ in self.traps:
            x, y = trap_pos // self.GRID_SIZE, trap_pos % self.GRID_SIZE
            self.canvas.create_rectangle(
                y * cell_size, x * cell_size,
                (y + 1) * cell_size, (x + 1) * cell_size,
                fill='black'
            )
        
        # Draw treasure (simple golden circle)
        treasure_x, treasure_y = self.treasure_pos // self.GRID_SIZE, self.treasure_pos % self.GRID_SIZE
        center_x = (treasure_y + 0.5) * cell_size
        center_y = (treasure_x + 0.5) * cell_size
        radius = cell_size * 0.3
        
        # Draw the golden circle
        self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill='gold',
            outline='orange'
        )
    
    def get_reward(self, state):
        if state == self.treasure_pos:
            return self.treasure_reward
        for trap_pos, penalty in self.traps:
            if state == trap_pos:
                return penalty
        return -1
    
    def move(self, state, action):
        x, y = state // self.GRID_SIZE, state % self.GRID_SIZE
        new_x, new_y = x, y
        
        if action == 'up' and x > 0:
            new_x = x - 1
        elif action == 'down' and x < self.GRID_SIZE-1:
            new_x = x + 1
        elif action == 'left' and y > 0:
            new_y = y - 1
        elif action == 'right' and y < self.GRID_SIZE-1:
            new_y = y + 1
            
        return new_x * self.GRID_SIZE + new_y
    
    def start_training(self):
        try:
            # Update parameters from GUI
            self.GRID_SIZE = int(self.grid_size_var.get())
            self.NUM_STATES = self.GRID_SIZE * self.GRID_SIZE
            self.EPISODES = int(self.episodes_var.get())
            self.treasure_pos = self.NUM_STATES - 1  # Always in bottom-right corner
            self.treasure_reward = int(self.treasure_reward_var.get())
            self.trap_penalty = int(self.trap_penalty_var.get())
            self.ALPHA = float(self.alpha_var.get())
            self.GAMMA = float(self.gamma_var.get())
            max_steps = int(self.max_steps_var.get())
            
            # Reset Q-table and debug information
            self.Q = np.zeros((self.NUM_STATES, self.NUM_ACTIONS))
            self.episode_rewards = []
            self.episode_steps = []
            self.successful_episodes = 0
            self.trap_hits = 0
            
            # Start training
            start_time = time.time()
            
            for episode in range(self.EPISODES):
                state = 0
                done = False
                episode_reward = 0
                steps = 0
                hit_trap = False
                
                while not done and steps < max_steps:  # Use user-defined max steps
                    # Epsilon-greedy action selection
                    if np.random.uniform(0, 1) < self.EPSILON:
                        action = np.random.choice(self.ACTIONS)
                    else:
                        action = self.ACTIONS[np.argmax(self.Q[state])]
                    
                    # Take action and observe result
                    next_state = self.move(state, action)
                    reward = self.get_reward(next_state)
                    done = (next_state == self.treasure_pos)
                    
                    # Check if hit trap
                    if next_state in [trap[0] for trap in self.traps]:
                        hit_trap = True
                    
                    # Q-learning update
                    old_value = self.Q[state, self.ACTIONS.index(action)]
                    next_max = np.max(self.Q[next_state])
                    new_value = (1 - self.ALPHA) * old_value + self.ALPHA * (reward + self.GAMMA * next_max)
                    self.Q[state, self.ACTIONS.index(action)] = new_value
                    
                    episode_reward += reward
                    state = next_state
                    steps += 1
                
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(steps)
                if done:
                    self.successful_episodes += 1
                if hit_trap:
                    self.trap_hits += 1
                
                # Update status every 100 episodes
                if (episode + 1) % 100 == 0:
                    success_rate = (self.successful_episodes / (episode + 1)) * 100
                    trap_rate = (self.trap_hits / (episode + 1)) * 100
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_steps = np.mean(self.episode_steps[-100:])
                    self.status_label.config(text=f"Episode {episode + 1}/{self.EPISODES}\n"
                                                f"Success Rate: {success_rate:.1f}%\n"
                                                f"Trap Rate: {trap_rate:.1f}%\n"
                                                f"Avg Reward: {avg_reward:.1f}\n"
                                                f"Avg Steps: {avg_steps:.1f}")
                    self.root.update()
            
            self.training_time = time.time() - start_time
            final_success_rate = (self.successful_episodes / self.EPISODES) * 100
            final_trap_rate = (self.trap_hits / self.EPISODES) * 100
            final_avg_reward = np.mean(self.episode_rewards)
            final_avg_steps = np.mean(self.episode_steps)
            
            status_message = f"Training completed in {self.training_time:.2f} seconds\n"
            status_message += f"Final Success Rate: {final_success_rate:.1f}%\n"
            status_message += f"Final Trap Rate: {final_trap_rate:.1f}%\n"
            status_message += f"Final Average Reward: {final_avg_reward:.1f}\n"
            status_message += f"Final Average Steps: {final_avg_steps:.1f}"
            
            self.status_label.config(text=status_message)
            messagebox.showinfo("Training Complete", status_message)
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for all parameters")
    
    def get_optimal_path(self):
        """Get the optimal path from start to goal"""
        path = []
        current_state = 0  # Start state
        visited = set()
        
        while current_state != self.treasure_pos and current_state not in visited:
            visited.add(current_state)
            path.append(current_state)
            
            # Get the best action for current state
            best_action_idx = np.argmax(self.Q[current_state])
            best_action = self.ACTIONS[best_action_idx]
            
            # Move to next state
            next_state = self.move(current_state, best_action)
            
            # If we can't move or we're stuck, break
            if next_state == current_state:
                break
                
            current_state = next_state
        
        # Add the final state (treasure) to the path if we reached it
        if current_state == self.treasure_pos:
            path.append(current_state)
        
        return path
    
    def show_results(self):
        self.canvas.delete("all")
        
        # Calculate cell size
        cell_size = 400 / self.GRID_SIZE
        
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, 400, fill='black')
            self.canvas.create_line(0, i * cell_size, 400, i * cell_size, fill='black')
        
        # Draw traps
        for trap_pos, _ in self.traps:
            x, y = trap_pos // self.GRID_SIZE, trap_pos % self.GRID_SIZE
            self.canvas.create_rectangle(
                y * cell_size, x * cell_size,
                (y + 1) * cell_size, (x + 1) * cell_size,
                fill='black'
            )
        
        # Draw treasure (simple golden circle)
        treasure_x, treasure_y = self.treasure_pos // self.GRID_SIZE, self.treasure_pos % self.GRID_SIZE
        center_x = (treasure_y + 0.5) * cell_size
        center_y = (treasure_x + 0.5) * cell_size
        radius = cell_size * 0.3
        
        # Draw the golden circle
        self.canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill='gold',
            outline='orange'
        )
        
        # Get and draw the optimal path
        optimal_path = self.get_optimal_path()
        for i in range(len(optimal_path) - 1):
            current_state = optimal_path[i]
            next_state = optimal_path[i + 1]
            
            # Calculate positions
            current_x, current_y = current_state // self.GRID_SIZE, current_state % self.GRID_SIZE
            next_x, next_y = next_state // self.GRID_SIZE, next_state % self.GRID_SIZE
            
            # Calculate centers
            current_center_x = (current_y + 0.5) * cell_size
            current_center_y = (current_x + 0.5) * cell_size
            next_center_x = (next_y + 0.5) * cell_size
            next_center_y = (next_x + 0.5) * cell_size
            
            # Draw arrow
            arrow_length = cell_size * 0.3
            if next_x < current_x:  # Up
                self.canvas.create_line(current_center_x, current_center_y, 
                                      current_center_x, current_center_y - arrow_length, 
                                      arrow=tk.LAST, fill='blue', width=2)
            elif next_x > current_x:  # Down
                self.canvas.create_line(current_center_x, current_center_y, 
                                      current_center_x, current_center_y + arrow_length, 
                                      arrow=tk.LAST, fill='blue', width=2)
            elif next_y < current_y:  # Left
                self.canvas.create_line(current_center_x, current_center_y, 
                                      current_center_x - arrow_length, current_center_y, 
                                      arrow=tk.LAST, fill='blue', width=2)
            else:  # Right
                self.canvas.create_line(current_center_x, current_center_y, 
                                      current_center_x + arrow_length, current_center_y, 
                                      arrow=tk.LAST, fill='blue', width=2)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    maze = MazeRL()
    maze.run()