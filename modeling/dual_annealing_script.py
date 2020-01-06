from ModelRunner import test_parameter_constraints, test_parameter_fitness, ModelType
from test_parameters import parameters60
from scipy.optimize import dual_annealing
import numpy as np
import time
import os
import sys

import pyximport
pyximport.install(language_level=3)

x_minima = []
start_time = time.time()

x_final_dir = "/home/jarpqd/data/dual_annealing"
os.system("mkdir {}".format(x_final_dir))
x_final_path = os.path.join(
    x_final_dir, "x_final_{}.csv".format(str(time.time())))


def evaluate(x):
    if not test_parameter_constraints(x, model_type=ModelType.full_model):
        return 9999
    return 1 - test_parameter_fitness(x, model_type=ModelType.full_model, use_constraints=True)


def callback(x, f, context):
    x_minima.append(list(x) + [f])
    x_final = np.asarray(x_minima)
    np.savetxt(x_final_path, x_final, delimiter=',')


def accept_test(f_new, x_new, f_old, x_old):
    return test_parameter_constraints(x_new)


parameters = parameters60
bounds = [(0.0, 0.0)] * len(parameters)
for i, p in enumerate(parameters):
    if p > 0:
        bounds[i] = (p / 4, p * 4)
    else:
        bounds[i] = (p * 4, p / 4)
bounds[56] = (0.1, 0.5)
bounds[57] = (0.5, 2.0)
bounds[58] = (0.5, 2.0)
bounds[59] = (0.1, 0.4)


x0 = parameters.copy()
niter = 50000

res = dual_annealing(func=evaluate, bounds=bounds,
                     callback=callback, x0=x0, maxfun=niter)

x_minima.append(list(res.x) + [res.fun])
x_final = np.asarray(x_minima)
np.savetxt(x_final_path, x_final, delimiter=',')
