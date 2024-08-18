import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import ast

class Maze:
    def __init__(self, maze, start_position, goal_position):
        # Initialize Maze object with the provided maze, start_position, and goal position
        self.maze = maze
        self.maze_height = len(maze) # Get the height of the maze (number of rows)
        self.maze_width = len(maze[0])  # Get the width of the maze (number of columns)
        self.start_position = start_position    # Set the start position in the maze as a tuple (x, y)
        self.goal_position = goal_position      # Set the goal position in the maze as a tuple (x, y)

    def show_maze(self):
        # Configure the plot
        fig, ax = plt.subplots()

        # Defining the row and col size
        row = self.maze_height
        col = self.maze_width
        grid_size = (col, row)

        # Defines the color of the girds as white
        color_map = np.ones((col, row, 3))

        # Finds the obstacle grids and make it black
        for x in range(col):
            for y in range(row):
                if self.maze[y][x] == 1:
                    color_map[y][x] = [0, 0, 0]  # Black for obstacles

        # Plot grid
        ax.imshow(color_map, origin='lower')

         # Plot the start and goal positions
        ax.scatter(self.start_position[0], self.start_position[1], color='red', label='Starting Position', s=100) 
        ax.scatter(self.goal_position[0], self.goal_position[1], color='cyan', label='Goal Position', s=100)  

        # Adding grid lines
        ax.set_xticks(np.arange(-.5, col, 1), minor=True)
        ax.set_yticks(np.arange(-.5, row, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(axis='both', which='both', length=0)
        plt.xticks(range(grid_size[0]))
        plt.yticks(range(grid_size[1]))
        
        # Grid labels
        plt.xticks([i for i in range(grid_size[0])], [str(i) for i in range(grid_size[0])])
        plt.yticks([i for i in range(grid_size[1])], [str(i) for i in range(grid_size[1])])

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        # Initialize the Q-learning agent with a Q-table containing all zeros
        # where the rows represent states, columns represent actions, and the third dimension is for each action (Up, Down, Left, Right)

        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 8)) # 8 actions: Up, Down, Left, Right and Diagonals
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
            return np.random.randint(8) # Choose a random action (index 0 to 3, representing Up, Down, Left, Right)
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
    

# This function simulates the agent's movements in the maze for a single episode.
def finish_episode(agent, maze, current_episode, actions, goal_reward=100, wall_penalty=-10, step_penalty=-1, train=True):
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
    path.append(maze.goal_position)
    return episode_reward, episode_step, path

def test_agent(agent, maze, actions, num_episodes=1):

    # This function evaluates an agent's performance in the maze. The function simulates the agent's movements in the maze,
    # updating its state, accumulating the rewards, and determining the end of the episode when the agent reaches the goal position.
    # The agent's learned path is then printed along with the total number of steps taken and the total reward obtained during the
    # simulation. The function also visualizes the maze with the agent's path marked in blue for better visualization of the
    # agent's trajectory.

    # Simulate the agent's behavior in the maze for the specified number of episodes

    episode_reward, episode_step, path = finish_episode(agent, maze, num_episodes, actions, train=False)


    # Print the learned path of the agent
    '''
    print("Learned Path:")
    for row, col in path:
        print(f"({row}, {col})-> ", end='')
    print("Goal!")
    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)
    '''
    # Visualize the maze using matplotlib
    fig, ax = plt.subplots()

    # Defining the row and col size
    row = maze.maze_height
    col = maze.maze_width
    grid_size = (col, row)

    # Defines the color of the girds as white
    color_map = np.ones((col, row, 3))

    # Finds the obstacle grids and make it black
    for x in range(col):
        for y in range(row):
            if maze.maze[y][x] == 1:
                color_map[y][x] = [0, 0, 0]  # Black for obstacles
            elif (x, y) in path:
                color_map[y][x] = [0, 1, 0]  # Green for path

    # Plot grid
    ax.imshow(color_map, origin='lower')

    # Plot the start and goal positions
    ax.scatter(maze.start_position[0], maze.start_position[1], color='red', label='Starting Position', s=100) 
    ax.scatter(maze.goal_position[0], maze.goal_position[1], color='cyan', label='Goal Position', s=100)  


    # Adding grid lines
    ax.set_xticks(np.arange(-.5, col, 1), minor=True)
    ax.set_yticks(np.arange(-.5, row, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(axis='both', which='both', length=0)
    plt.xticks(range(grid_size[0]))
    plt.yticks(range(grid_size[1]))
    
    # Grid labels
    plt.xticks([i for i in range(grid_size[0])], [str(i) for i in range(grid_size[0])])
    plt.yticks([i for i in range(grid_size[1])], [str(i) for i in range(grid_size[1])])

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    return episode_step, episode_reward
    
def train_agent(agent, maze, actions, num_episodes=100):
    # Lists to store the data for plotting
    episode_rewards = []
    episode_steps = []

    # Loop over the specified number of episodes
    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, actions, train=True)

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
    print(f"The last reward is: {episode_rewards.pop()}")

    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, max(episode_steps))
    plt.title('Steps per Episode')

    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The last step is: {episode_steps.pop()}")

    plt.tight_layout()
    plt.show()

def main():
    # Variable definitions
    start_grid_coor = (0,0)
    end_grid_coor = (15,11)
    planned_path = None
    num_of_ep = 1000

    maze_layout = np.array([
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0],
   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])
    action = [(0, 1), (0, -1), (1, 0), (-1, 0),(1, 1), (1, -1), (-1, 1), (-1, -1)]


    # Objects
    maze = Maze(maze_layout, start_grid_coor, end_grid_coor)
    agent = QLearningAgent(maze)

    # Run code
    maze.show_maze()

    train_agent(agent, maze, action, num_episodes=num_of_ep)
    test_agent(agent, maze, action, num_episodes=num_of_ep)
    
main()