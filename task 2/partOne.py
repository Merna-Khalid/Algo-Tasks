import numpy as np
import random
from math import sqrt, sin

def exhaustive(f, a, b, e=0.001):
	n = int((b - a) / e)
	mini = float('inf') 
	ind = 1
	for i in range(1, n + 1):
		x_m = a + i * (b - a) / n
		if mini > f(x_m):
			mini = f(x_m)
			ind = i
	return a + ind * (b - a) / n

def dichotomy(f, a, b, e=0.001):
	l = random.uniform(0.0000001, e)
	iters = 0
	while abs(a - b) >= e:
		x1 = (a + b - l) / 2.0
		x2 = (a + b + l) / 2.0
		if f(x1) <= f(x2):
			b = x2
		else:
			a = x1
		iters += 1
	return a, b, iters
	

def golden_section(f, a, b, e=0.001):
	x1 = a + (3 - sqrt(5)) / 2.0 * (b - a)
	x2 = b + (sqrt(5) - 3) / 2.0 * (b - a)
	iters = 1
	if f(x1) <= f(x2):
		b, x2 = x2, x1
		while abs(a - b) >= e:
			x1 = a + (3 - sqrt(5)) / 2.0 * (b - a)
			b, x2 = x2, x1
			iters += 1
	else:
		a, x1 = x1, x2
		while abs(a - b) >= e:
			x2 = b + (sqrt(5) - 3) / 2.0 * (b - a)
			a, x1 = x1, x2
			iters += 1
	return a, b, iters



def cube(x):
	return x ** 3

def secondF(x):
	return abs(x - 0.2)

def thirdF(x):
	return x * sin(1/float(x))
#print(exhaustive(cube, 0, 1, 0.001))

"""
a, b, iters = dichotomy(cube, 0, 1, 0.001)
print(a, cube(a))
print(b, cube(b))
print(iters)
"""
"""
a, b, iters = dichotomy(secondF, 0, 1, 0.001)
print(a, secondF(a))
print(b, secondF(b))
print(iters)

a, b, iters = golden_section(secondF, 0, 1, 0.001)
print(a, secondF(a))
print(b, secondF(b))
print(iters)
"""
