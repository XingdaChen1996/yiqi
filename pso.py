import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pso(obj_fun, bounds, num_particles=100, max_iterations=100, w=0.5, c1=1, c2=2, image_path=None, seed=0):
    np.random.seed(seed)
    dim = len(bounds)
    # Initialize the particles
    particles_position = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_particles, dim))
    particles_velocity = np.zeros((num_particles, dim))
    particles_best_position = particles_position.copy()
    particles_best_value = np.array([obj_fun(x) for x in particles_position])

    # Initialize the global best position
    global_best_index = np.argmin(particles_best_value)
    global_best_position = particles_best_position[global_best_index]
    global_best_value = particles_best_value[global_best_index]

    history_best_value = np.full(max_iterations, np.inf)
    # Run the algorithm
    for i in range(max_iterations):
        # Update the particles velocity and position
        r1 = np.random.uniform(size=(num_particles, dim))
        r2 = np.random.uniform(size=(num_particles, dim))
        particles_velocity = w * particles_velocity \
                             + c1 * r1 * (particles_best_position - particles_position) \
                             + c2 * r2 * (global_best_position - particles_position)
        particles_position = particles_position + particles_velocity

        # Apply the bounds
        particles_position = np.clip(particles_position, bounds[:, 0], bounds[:, 1])

        # Update the particles best position and value
        new_particles_best_value = np.array([obj_fun(x) for x in particles_position])
        mask = new_particles_best_value < particles_best_value
        particles_best_position[mask] = particles_position[mask]
        particles_best_value[mask] = new_particles_best_value[mask]

        # Update the global best position and value
        global_best_index = np.argmin(particles_best_value)
        global_best_position = particles_best_position[global_best_index]
        global_best_value = particles_best_value[global_best_index]

        history_best_value[i] = global_best_value
        # Print the best value in the current iteration
        print('Iteration {}: Best value = {:.4f}'.format(i + 1, global_best_value))

    if image_path is not None:
        pd.Series(history_best_value).plot()
        plt.title("PSO optimization")
        plt.xlabel("iteration")  # 设置横坐标信息
        plt.ylabel("fitness")  # 设置纵坐标信息
        plt.savefig(image_path)
        plt.close()

    return global_best_position, global_best_value


if __name__ == '__main__':
    def fun1(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def fun2(x):
        n = len(x)
        return 10 * n + sum([xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

    bounds = np.array([(5, 50), (-10, 10), (-10, 10)])
    num_particles = 100
    max_iterations = 100
    image_path = '1.png'
    solution, fitness = pso(obj_fun=fun1,
                            bounds=bounds,
                            num_particles=num_particles,
                            max_iterations=max_iterations,
                            image_path=image_path)
