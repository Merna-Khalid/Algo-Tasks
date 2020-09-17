import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_f(x, w):
	return w[0] * x + w[1]

def linear_f1(x, a, b):
	return np.array([a * i + b for i in x])

def linear_f2(x, w):
	return np.array([w[0] * i + w[1] for i in x])

def rational_f(x, w):
	return w[0] / (1 + w[1] * x)

def rational_f1(x, a, b):
	return np.array([a / (1 + b * i) for i in x])

def rational_f2(x, w):
	return np.array([w[0] / (1 + w[1] * i) for i in x])


# x, y -> numpy arrays

def gradient(x, y, w, f):
	e = y - f(x, w)
	return np.array([-2* x * e, -2 * e]).sum(axis=1)


def error(x, y, w):
	return 1/len(x) * np.sum(np.square(linear_f2(x, w) - y))

def linear_reg(x, y, lr=0.001):
	e = 999
	w = [random.uniform(0, 1), random.uniform(0, 1)]
	iters = 0
	iter_max = 300
	while iters < iter_max and e > 0.0001:
		gr = gradient(x, y, w, linear_f2)
		w -= lr * gr
		iters += 1
		e = error(x, y, w)
	return w

def rational_reg(x, y):
	w, _ = curve_fit(rational_f1, x, y)
	return w

"""
a = random.uniform(0, 1)
b = random.uniform(0, 1)
x = []
y = []
n = 20
for i in range(0, n):
	l = np.random.normal(0.5, 0.01)
	x.append(i / n)
	y.append(a * i / n + b + l)

x = np.array(x)
y = np.array(y)	
w = [1, 1]

#w = rational_reg(x, y)
w, _ = curve_fit(rational_f, x, y)

print(w)
print(a, b)
print(error(x, y, w))

#print(rational_f(x, w))
plt.plot(x, rational_f(x, w[0], w[1]), color="red")
plt.plot(x, y)
plt.show()
"""


