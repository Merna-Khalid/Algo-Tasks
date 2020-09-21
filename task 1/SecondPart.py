import numpy as np
import matplotlib.pyplot as plt
import functools
import random
import timeit
from scipy.optimize import curve_fit

def measure_avg_time(fun, size):
	a = np.array([[random.uniform(0, 1) for i in range(size)] for j in range(size)])
	b = np.array([[random.uniform(0, 1) for i in range(size)] for j in range(size)])
	return timeit.timeit(functools.partial(fun, size, a, b), number=5) / 5.0

def matrix_multi(size, a, b):
	return a.dot(b)

def n_2(x, a, b, c):
	return a * np.square(x) + b * x + c
	
timing = []

for i in range(1, 1001):
	timing.append(measure_avg_time(matrix_multi, i))

l1, = plt.plot(range(1, 1001), timing)
coefs, _ = curve_fit(n_2, range(1, 1001), timing)
l2, = plt.plot(range(1, 1001), [n_2(i, coefs[0], coefs[1], coefs[2]) for i in range(1, 1001)])
plt.title("Matrix multiplication function")
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()	
