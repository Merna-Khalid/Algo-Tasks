import numpy as np
import timeit
import random
import matplotlib.pyplot as plt
import functools
import QuickSort as qs

def measure_avg_time(fun, size):
	x = [random.uniform(0, 1) for i in range(size)]
	return timeit.timeit(functools.partial(fun, size, x), number=5) / 5.0	


timing = []

def const_func(size, x):
	for i in range(size):
		x[i] += 1

for i in range(1, 2001):
	timing.append(measure_avg_time(const_func, i))


plt.plot(range(1, 2001), timing, )
plt.title("constant function")
plt.show()


timing = []

def sum_func(size, x):
	return sum(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(sum_func, i))


plt.plot(range(1, 2001), timing)
plt.title("sum function")
plt.show()


timing = []

def multi_func(size, x):
	return np.prod(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(multi_func, i))


plt.plot(range(1, 2001), timing)
plt.title("multi function")
plt.show()

timing = []

def poly(n, x, v=1.5):
	s = 0
	for i in range(n - 1):
		s += x[i] * np.power(v, i)
	return s

def poly_func(size, x):
	return poly(size, x)

for i in range(1, 2001):
	timing.append(measure_avg_time(poly_func, i))


plt.plot(range(1, 2001), timing)
plt.title("poly function")
plt.show()


timing = []

def horner(n, x, v=1.5):
	s = 0
	for i in range(n - 1):
		s += x[i] + s * v
	return s

def horner_func(size, x):
	return horner(size, x)

for i in range(1, 2001):
	timing.append(measure_avg_time(horner_func, i))


plt.plot(range(1, 2001), timing)
plt.title("horner function")
plt.show()

"""
timing = []

def bubble(n, x, v=1.5):
	for i in range(n):
		for j in range(i, 1, -1):
			if x[j] < x[j - 1]:
				x[j], x[j - 1] = x[j - 1], x[j]

def bubble_func(size, x):
	return bubble(size, x)

for i in range(1, 2001):
	timing.append(measure_avg_time(bubble_func, i))
	print(i)


plt.plot(range(1, 2001), timing)
plt.title("bubble sort function")
plt.show()
"""

timing = []

def quick_func(size, x):
	return qs.quicksort(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(quick_func, i))


plt.plot(range(1, 2001), timing)
plt.title("quick sort function")
plt.show()