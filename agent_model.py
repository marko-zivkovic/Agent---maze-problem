import numpy as np
import matplotlib.pyplot as plt
import time


class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze_layout.shape[0]
        self.maze_width = maze_layout.shape[1]
        self.start_position = start_position
        self.goal_position = goal_position

    def show_maze(self):
        plt.figure(figsize=(5,5))
        plt.imshow(self.maze, cmap = 'gray')
        plt.text(self.start_position[0], self.start_position[1], '+', ha='center', va='center', color='red', fontsize=25)
        plt.text(self.goal_position[0], self.goal_position[1], '+', ha='center', va='center', color='yellow', fontsize=25)
        plt.xticks([]), plt.yticks([])
        plt.show()

class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        # Initialize the Q-learning agent with a Q-table containing all zeros
        # where the rows represent states, columns represent actions, and the third dimension is for each action (Up, Down, Left, Right)
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4)) # 4 actions: Up, Down, Left, Right
        self.learning_rate = learning_rate          # Learning rate controls how much the agent updates its Q-values after each action
        self.discount_factor = discount_factor      # Discount factor determines the importance of future rewards in the agent's decisions
        self.exploration_start = exploration_start  # Exploration rate determines the likelihood of the agent taking a random action
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
        # Calculate the current exploration rate using the given formula
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate

    def get_action(self, state, current_episode): # State is tuple representing where agent is in maze (x, y)
        exploration_rate = self.get_exploration_rate(current_episode)
        # Select an action for the given state either randomly (exploration) or using the Q-table (exploitation)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4) # Choose a random action (index 0 to 3, representing Up, Down, Left, Right)
        else:
            return np.argmax(self.q_table[state]) # Choose the action with the highest Q-value for the given state

    def update_q_table(self, state, action, next_state, reward):
        # Find the best next action by selecting the action that maximizes the Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state])
        # Get the current Q-value for the current state and action
        current_q_value = self.q_table[state][action]
        # Q-value update using Q-learning formula
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)
        # Update the Q-table with the new Q-value for the current state and action
        self.q_table[state][action] = new_q_value    

#////////////////////////////////////////////////////parametri

def load_matrix(matrix_path, ROWS):
    niz_brojeva = []
    matrica = []
    with open(matrix_path, 'r') as file:
        matrix = [list(map(int, line.strip().split(','))) for line in file.readlines()]
    for i in range(ROWS):
       for j in range(ROWS):
           broj = matrix[i][j]
           niz_brojeva.append(broj)
       matrica.append(niz_brojeva)
       niz_brojeva = []
    
    return np.array(matrica)
# pocetak i kraj //////////////////////////////////////////////////////////////////////////////
path = 'dataset\\train\\data_y\\maze 7.txt'
size = 21 #21x21
maze_layout = load_matrix(path,size)
maze = Maze(maze_layout, (0, 1), (20, 19))
maze.show_maze()
#AKCIJE
actions = [(-1, 0), # Up
          (1, 0),   # Down
          (0, -1),  # Left
          (0, 1)]   # Right
#NAGRADE
goal_reward = 1000
wall_penalty = -10
step_penalty = -1
#agent ////////////////////////////////////////////////////////////////////////////////////////
# This function simulates the agent's movements in the maze for a single episode.
def finish_episode(agent, maze, current_episode, train=True):
    # Initialize the agent's current state to the maze's start position
    current_state = maze.start_position
    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]
    # Continue until the episode is done
    while not is_done:
        # Get the agent's action for the current state using its Q-table
        action = agent.get_action(current_state, current_episode)
        # Compute the next state based on the chosen action
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])
        # Check if the next state is out of bounds or hitting a wall
        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[1]][next_state[0]] == 1:
            reward = wall_penalty
            next_state = current_state
        # Check if the agent reached the goal:
        elif next_state == (maze.goal_position):
            path.append(current_state)
            reward = goal_reward
            is_done = True
        # The agent takes a step but hasn't reached the goal yet
        else:
            path.append(current_state)
            reward = step_penalty
        # Update the cumulative reward and step count for the episode
        episode_reward += reward
        episode_step += 1
        # Update the agent's Q-table if training is enabled
        if train == True:
            agent.update_q_table(current_state, action, next_state, reward)
        # Move to the next state for the next iteration
        current_state = next_state
    # Return the cumulative episode reward, total number of steps, and the agent's path during the simulation
    return episode_reward, episode_step, path

# This function evaluates an agent's performance in the maze. The function simulates the agent's movements in the maze,
# updating its state, accumulating the rewards, and determining the end of the episode when the agent reaches the goal position.
# The agent's learned path is then printed along with the total number of steps taken and the total reward obtained during the
# simulation. The function also visualizes the maze with the agent's path marked in blue for better visualization of the
# agent's trajectory.
def test_agent(agent, maze, num_episodes):
    # Simulate the agent's behavior in the maze for the specified number of episodes
    print(f"testiranje agenta")
    episode_reward, episode_step, path = finish_episode(agent, maze, num_episodes, train=False)
    print(f"finish_ep agenta")
    # Print the learned path of the agent
    print("Learned Path:")
    for row, col in path:
        print(f"({row}, {col})-> ", end='')
    print("Goal!")
    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)
    # Clear the existing plot if any
    if plt.gcf().get_axes():
        plt.cla()
    # Visualize the maze using matplotlib
    plt.figure(figsize=(5,5))
    plt.imshow(maze.maze, cmap='gray')
    # Mark the start position (red 'S') and goal position (green 'G') in the maze
    plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal_position[0], maze.goal_position[1], 'F', ha='center', va='center', color='green', fontsize=20)
    # Mark the agent's path with blue '#' symbols
    for position in path:
        plt.text(position[0], position[1], "*", va='center', color='red', fontsize=20)
    # Remove axis ticks and grid lines for a cleaner visualization
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()
    return episode_step, episode_reward

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#TEST
agent = QLearningAgent(maze)
test_agent(agent, maze, 1)
#TRAIN
def train_agent(agent, maze, num_episodes):
    # Lists to store the data for plotting
    episode_rewards = []
    episode_steps = []

    # Loop over the specified number of episodes
    print(f"treniranje agenta")
    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)

        # Store the episode's cumulative reward and the number of steps taken in their respective lists
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

    # Plotting the data after training is completed
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')
    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 1000)
    plt.title('Steps per Episode')
    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")
    plt.tight_layout()
    plt.show()

# Training the agent
train_agent(agent, maze, num_episodes=700)
# Testing the agent after training
test_agent(agent, maze, num_episodes=700)
print("projekat je gotov")