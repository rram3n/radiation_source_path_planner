import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import heapq

# Define the Cell class
class Cell:
    def __init__(self):
      # Parent cell's row index
        self.parent_i = 0
    # Parent cell's column index
        self.parent_j = 0
    # Total cost of the cell (g + h)
        self.f = float('inf')
    # Cost from start to this cell
        self.g = float('inf')
    # Heuristic cost from this cell to destination
        self.h = 0

class AStar:
    def __init__(self):
        self.grid_row = 20
        self.grid_col = 20

    # Check if a cell is valid (within the grid)
    def is_valid(self, row, col):
        return (row >= 0) and (row < self.grid_row) and (col >= 0) and (col < self.grid_col)

    # Check if a cell is unblocked
    def is_unblocked(self, grid, row, col):
        return grid[row][col] == 0

    # Check if a cell is the destination
    def is_destination(self, row, col, dest):
        return row == dest[1] and col == dest[0]

    # Calculate the heuristic value of a cell (Euclidean distance to destination)
    def calculate_h_value(self, row, col, dest):
        return ((row - dest[1]) ** 2 + (col - dest[0]) ** 2) ** 0.5

    # Trace the path from source to destination
    def trace_path(self, cell_details, dest):
        #print("The Path is ")
        path = []
        row = dest[1]
        col = dest[0]

        # Trace the path from destination to source using parent cells
        while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
            path.append((col, row))
            temp_row = cell_details[row][col].parent_i
            temp_col = cell_details[row][col].parent_j
            row = temp_row
            col = temp_col

        # Add the source cell to the path
        path.append((col, row))
        # Reverse the path to get the path from source to destination
        path.reverse()

        # Print the path
        #for i in path:
            #print("->", i, end=" ")
        #print()
        return path

    # Implement the A* search algorithm
    def a_star_search(self, grid, src, dest):
        # Check if the source and destination are valid
        if not self.is_valid(src[1], src[0]) or not self.is_valid(dest[1], dest[0]):
            print("Source or destination is invalid")
            return

        # Check if the source and destination are unblocked
        if not self.is_unblocked(grid, src[1], src[0]) or not self.is_unblocked(grid, dest[1], dest[0]):
            print("Source or the destination is blocked")
            return

        # Check if we are already at the destination
        if self.is_destination(src[1], src[0], dest):
            print("We are already at the destination")
            return

        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.grid_col)] for _ in range(self.grid_row)]
        # Initialize the details of each cell
        cell_details = [[Cell() for _ in range(self.grid_col)] for _ in range(self.grid_row)]

        # Initialize the start cell details
        i = src[1]
        j = src[0]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i = i + dir[1]
                new_j = j + dir[0]

                # If the successor is valid, unblocked, and not visited
                if self.is_valid(new_i, new_j) and self.is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if self.is_destination(new_i, new_j, dest):
                        # Set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        #print("The destination cell is found")
                        # Trace and print the path from source to destination
                        path = self.trace_path(cell_details, dest)
                        found_dest = True
                        return path
                    else:
                        # Calculate the new f, g, and h values
                        g_new = cell_details[i][j].g + 1.0
                        h_new = self.calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new

                        # If the cell is not in the open list or the new f value is smaller
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")

    def plot_grid(self, grid, path, start_pos, goal_pos, source_pos):
        fig, ax = plt.subplots()
        row = len(grid)
        col = len(grid[0])
        grid_size = (col, row)

        color_map = np.ones((col, row, 3))

        for x in range(col):
            for y in range(row):
                if (x, y) in path:
                    color_map[y][x] = [0, 1, 0]  # Green for path
                elif grid[y][x] == 1:
                    color_map[y][x] = [0, 0, 0]  # Black for obstacles

        ax.imshow(color_map, origin='lower')

         # Highlight the start and terminal positions
        ax.scatter(start_pos[0], start_pos[1], color='red', label='Current Robot Position', s=100) 
        ax.scatter(goal_pos[0], goal_pos[1], color='cyan', label='Estimated Goal', s=100)  
        ax.scatter(source_pos[0], source_pos[1], color='green', label='Source', s=100)  

        # Adding grid lines for clarity
        ax.set_xticks(np.arange(-.5, col, 1), minor=True)
        ax.set_yticks(np.arange(-.5, row, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.tick_params(axis='both', which='both', length=0)
        plt.xticks(range(grid_size[0]))
        plt.yticks(range(grid_size[1]))
        
        # Set grid labels
        plt.xticks([i for i in range(grid_size[0])], [str(i) for i in range(grid_size[0])])
        plt.yticks([i for i in range(grid_size[1])], [str(i) for i in range(grid_size[1])])

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

class ParticleFilter:
    def __init__(self, start_coor):
        self.num_particles = 1000
        self.width = 20
        self.height = 20
        self.source_position = [18,7]
        self.neff_threshold = 20
        self.robot_position = start_coor
        self.estimated_goal = None
        self.iteration = 10

        self.particles = self.initialize_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Gaussian plume variables
        self.Q = 1e6  # Emission rate
        self.u = 1  # Wind speed
        self.sigma_x = 150  # Dispersion coefficient in the x-direction
        self.sigma_y = 150  # Dispersion coefficient in the y-direction

        # Visualsation
        self.path = np.array([self.robot_position])
        self.old_path = None

    def gaussian_plume(self, x, y, x0, y0):
        """ Calculate the Gaussian plume concentration at a point (x, y). """
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        if distance == 0:
            return float('inf')  # Avoid division by zero at the source point
        exp_part = np.exp(-((x - x0)**2) / (2 * self.sigma_x**2) - ((y - y0)**2) / (2 * self.sigma_y**2))
        return (self.Q / (2 * math.pi * self.u * self.sigma_y * self.sigma_x)) * exp_part

    def calc_robot_current_intensity(self):
        intensity = self.gaussian_plume(self.robot_position[0], self.robot_position[1], self.source_position[0], self.source_position[1])
        #print("Robot Current Intensity:", intensity)
        return intensity

    def initialize_particles(self):
        ''' Initialize particles to random positions within the defined boundary. '''
        particles = np.empty((self.num_particles, 2))
        particles[:, 0] = np.random.uniform(0, self.width, self.num_particles)
        particles[:, 1] = np.random.uniform(0, self.height, self.num_particles)
        return particles

    def predict(self, move_dist=1):
        ''' Predict the next state of the particles based on the previous state with random movements. '''
        movements = np.random.randn(self.num_particles, 2) * move_dist
        self.particles += movements
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.height)

    def update(self):
        ''' Update particle weights based on the measured concentration values. '''
        robot_current_intensity = self.calc_robot_current_intensity()
        for i in range(self.num_particles):
            x, y = self.particles[i]
            predicted_intensity = self.gaussian_plume(self.robot_position[0], self.robot_position[1], x, y) # pretend if the particle is the source of a gaussian plume
            weight = abs(predicted_intensity - robot_current_intensity) # if the predicted intensity if close to robot's actual reading, then it has a higher likelihood of being the source
            self.weights[i] = 1/weight
        self.weights /= np.sum(self.weights)  # Normalize
        #print('sum:',sum(self.weights))
        #print('weights:',self.weights)

    def resample(self):
        ''' Resample particles according to their weights to focus on high probability areas. '''
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure the sum of weights is exactly one
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))

        # resample according to indexes
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def mcmc_move(self, std_dev=0.1):
        """
        Apply a Markov Chain Monte Carlo move to each particle to introduce small random changes.
        std_dev: Standard deviation of the Gaussian noise added for the MCMC move.
        """
        noise = np.random.normal(0, std_dev, size=(self.num_particles, 2))
        self.particles += noise
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.height)

    def estimate(self):
        ''' Estimate the current position of the source based on the particles. '''
        mean = np.average(self.particles, weights=self.weights, axis=0)
        self.estimated_goal = mean
        return mean

    def move_to_goal(self, path):
        '''
        step_size = 10  # Define step size
        direction = self.estimated_goal - self.robot_position
        direction /= np.linalg.norm(direction)
        self.robot_position += direction * step_size
        self.path = np.vstack([self.path, self.robot_position.copy()]) # Append new position
        #print(self.path)
        '''
        #print(path[0][0])
        path.pop(0)
        print('current robot pos:', self.robot_position)
        print(path)
        self.robot_position[0] = path[0][0]
        self.robot_position[1] = path[0][1]
        self.path = np.vstack([self.path, self.robot_position.copy()]) # Append new position

        path.pop(0)
        self.old_path = path

        print('new robot pos:', self.robot_position)
        return self.robot_position
    
    def continue_with_old_path(self):
        self.robot_position[0] = self.old_path[0][0]
        self.robot_position[1] = self.old_path[0][1]
        self.path = np.vstack([self.path, self.robot_position.copy()]) # Append new position
        self.old_path.pop(0)
        print('planned path:', self.old_path)

    def plot(self):
        ''' Plot the particles and the estimated source position. '''
        plt.figure(figsize=(10, 8))

        #print(self.particles)
        scatter = plt.scatter(self.particles[:, 0], self.particles[:, 1], color='k', s=5, label='Particles')
        #print(max(self.weights), min(self.weights))
        #plt.colorbar(scatter, label='Particle Intensity')
        
        plt.scatter(self.estimated_goal[0], self.estimated_goal[1], color='blue', s=100, label='Estimated Position')
        plt.scatter(*self.source_position, color='green', s=100, label='True Source', marker='x')
        plt.scatter(*self.robot_position, color='red', s=100, label='Current Robot Position')

        path_x = []
        path_y = []
        for i, coor in enumerate(self.path):
            path_x.append(coor[0])
            path_y.append(coor[1])

        plt.plot(path_x, path_y, color='green', alpha=0.7, label='Path')

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.title('Particle Filter with Gaussian Plume Model')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

def main():
    # Variable definitions
    start_grid_coor = [2,2]
    current_gird_coor = start_grid_coor
    planned_path = None

    # Objects
    a_star = AStar()
    pf = ParticleFilter(start_grid_coor)

    grid = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]  
    
    # Variable definitions
    estimated_position_int = [0,0]

    print('starting pf')
    for i in range(pf.iteration+10):
        print('iteration: ', i)

        pf.update()  # Update based on a measured value

        neff = pf.neff()
        if neff > pf.neff_threshold:
            print('resampling')
            pf.resample()  # Resample according to weights
            pf.mcmc_move() # apply Markov Chain Monte Carlo move
        
        estimated_position = pf.estimate()
        estimated_position_int[0] = round(estimated_position[0])
        estimated_position_int[1] = round(estimated_position[1])

        planned_path = a_star.a_star_search(grid, current_gird_coor, estimated_position_int)

        print("current robot position:", current_gird_coor)
        print("location of source:", pf.source_position)
        print("location of estimated source:", estimated_position)
        #print('estimated position int:', estimated_position_int)
        print('planned path:',planned_path)

        if planned_path is None:
            print('the predicted source is in an obstacle, using old path')
            pf.continue_with_old_path()
            a_star.plot_grid(grid, pf.old_path, current_gird_coor, estimated_position_int, pf.source_position)
            
        else:
            print('new estimation found, creating new path')
            pf.move_to_goal(planned_path)
            a_star.plot_grid(grid, planned_path, current_gird_coor, estimated_position_int, pf.source_position)

        
        pf.plot()  # Plot particle filter
        
        current_gird_coor = pf.robot_position

main()