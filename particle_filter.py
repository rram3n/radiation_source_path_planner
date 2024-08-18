import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

class ParticleFilter:
    def __init__(self):
        self.num_particles = 1000
        self.width = 200
        self.height = 200
        self.source_position = [50,145]
        self.neff_threshold = 20
        self.robot_position = [0,0]
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

    def gaussian_plume(self, x, y, x0, y0):
        """ Calculate the Gaussian plume concentration at a point (x, y). """
        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
        if distance == 0:
            return float('inf')  # Avoid division by zero at the source point
        exp_part = np.exp(-((x - x0)**2) / (2 * self.sigma_x**2) - ((y - y0)**2) / (2 * self.sigma_y**2))
        return (self.Q / (2 * math.pi * self.u * self.sigma_y * self.sigma_x)) * exp_part

    def calc_robot_current_intensity(self):
        intensity = self.gaussian_plume(self.robot_position[0], self.robot_position[1], self.source_position[0], self.source_position[1])
        print("Robot Current Intensity:", intensity)
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

    def move_to_goal(self):
        step_size = 10  # Define step size
        direction = self.estimated_goal - self.robot_position
        direction /= np.linalg.norm(direction)
        self.robot_position += direction * step_size
        self.path = np.vstack([self.path, self.robot_position.copy()]) # Append new position
        #print(self.path)


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
    # Example usage:
    pf = ParticleFilter()
    print('starting pf')
    for i in range(pf.iteration):
        print('iteration: ', i)
        pf.update()  # Update based on a measured value
        neff = pf.neff()
        if neff > pf.neff_threshold:
            print('resampling')
            pf.resample()  # Resample according to weights
            pf.mcmc_move() # apply Markov Chain Monte Carlo move
        estimated_position = pf.estimate()
        print("location of source:", pf.source_position)
        print("location of estimated source:", estimated_position)
        pf.plot()  # Plot the results
        pf.move_to_goal()

main()