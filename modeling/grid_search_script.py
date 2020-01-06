import pyximport; pyximport.install(language_level=3)

from test_parameters import parameters2
from ModelRunner import test_parameter_fitness

from multiprocessing import Pool
import numpy as np


grid_size = 2
parameters_grid = (
    np.linspace(start=(parameters2[0] / 4), stop=(parameters2[0] * 4), num=grid_size),
    np.linspace(start=(parameters2[1] / 4), stop=(parameters2[1] * 4), num=grid_size)
)
results_grid = np.zeros((grid_size, grid_size))

grid_points = np.transpose([np.tile(parameters_grid[0], len(parameters_grid[1])),
                            np.repeat(parameters_grid[1], len(parameters_grid[0]))])



agents = 5
chunksize = 3
with Pool(processes=agents) as pool:
    results = pool.map(test_parameter_fitness, grid_points, chunksize)


with open("grid_results.csv", 'w') as file:
    for (p1, p2), result in zip(grid_points, results):
        file.write("{},{},{}\n".format(p1, p2, result))