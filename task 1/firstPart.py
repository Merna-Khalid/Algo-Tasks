import numpy as np
import timeit
import random
import matplotlib.pyplot as plt
import functools
import QuickSort as qs
from scipy.optimize import curve_fit

def measure_avg_time(fun, size):
	x = [random.uniform(0, 1) for i in range(size)]
	return timeit.timeit(functools.partial(fun, size, x), number=5) / 5.0	

def lin(x, a, b):
	return a * x + b

def nlogn(x, a, b):
	return a * x * np.log(x) + b

def n_2(x, a, b, c):
	return a * np.square(x) + b * x + c

timing = []

def const_func(size, x):
	x[0] += 1

for i in range(1, 2001):
	timing.append(measure_avg_time(const_func, i))


l1, = plt.plot(range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [timing[0]] * 2000)
plt.title("constant function")
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()


timing = []

def sum_func(size, x):
	return sum(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(sum_func, i))


l1, = plt.plot(range(1, 2001), timing)
coefs, _ = curve_fit(lin, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [lin(i, coefs[0], coefs[1]) for i in range(1, 2001)])
plt.title("sum function")
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()


timing = []

def multi_func(size, x):
	return np.prod(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(multi_func, i))


l1, = plt.plot(range(1, 2001), timing)
coefs, _ = curve_fit(lin, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [lin(i, coefs[0], coefs[1]) for i in range(1, 2001)])
plt.title("multi function")
plt.legend((l1, l2), ("experimental", "theoretical"))
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


l1, = plt.plot(range(1, 2001), timing)
coefs, _ = curve_fit(n_2, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [n_2(i, coefs[0], coefs[1], coefs[2]) for i in range(1, 2001)])
plt.title("poly function")
plt.legend((l1, l2), ("experimental", "theoretical"))
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


l1, = plt.plot(range(1, 2001), timing)
coefs, _ = curve_fit(lin, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [lin(i, coefs[0], coefs[1]) for i in range(1, 2001)])
plt.title("horner function")
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()


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


l1, = plt.plot(range(1, 2001), timing)
coefs, _ = curve_fit(n_2, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [n_2(i, coefs[0], coefs[1], coefs[2]) for i in range(1, 2001)])
plt.title("bubble sort function")
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()


timing = []

def quick_func(size, x):
	return qs.quicksort(x)

for i in range(1, 2001):
	timing.append(measure_avg_time(quick_func, i))


l1, = plt.plot(range(1, 2001), timing)
plt.title("quick sort function")
coefs, _ = curve_fit(nlogn, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [nlogn(i, coefs[0], coefs[1]) for i in range(1, 2001)])
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()


timing = []

def tim_func(size, x):
	return x.sort()

for i in range(1, 2001):
	timing.append(measure_avg_time(tim_func, i))


l1, = plt.plot(range(1, 2001), timing)
plt.title("tim sort function")
coefs, _ = curve_fit(nlogn, range(1, 2001), timing)
l2, = plt.plot(range(1, 2001), [nlogn(i, coefs[0], coefs[1]) for i in range(1, 2001)])
plt.legend((l1, l2), ("experimental", "theoretical"))
plt.show()