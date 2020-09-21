import numpy as np
import random
from math import sqrt, sin

# returns minimum, number of f-calculations, number of iterations
def exhaustive(f, a, b, e=0.001):
	n = int((b - a) / e)
	mini = float('inf') 
	ind = 1
	for i in range(1, n + 1):
		x_m = a + i * (b - a) / n
		if mini > f(x_m):
			mini = f(x_m)
			ind = i
	return a + ind * (b - a) / n, n, n

# returns a, b where minimum is between, number of f-cals, iterations
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
	return a, b, iters * 2, iters
	

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
	return a, b, 2, iters



def cube(x):
	return x ** 3

def secondF(x):
	return abs(x - 0.2)

def thirdF(x):
	return x * sin(1/float(x))


print("cube function:")
print("\texhaustives search :")
print("\t  minimum, f_cals, iters")
print("\t",exhaustive(cube, 0, 1, 0.001))


a, b, fcals, iters = dichotomy(cube, 0, 1, 0.001)
print("\tdichotomy :")
print("\t\ta/f(a): ",a, cube(a))
print("\t\tb/f(b): ",b, cube(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)


a, b, fcals, iters = golden_section(cube, 0, 1, 0.001)
print("\tgolden section :")
print("\t\ta/f(a): ",a, cube(a))
print("\t\tb/f(b): ",b, cube(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)


#---------------------------------------------------------------

print("second function:")
print("\texhaustives search :")
print("\t  minimum, f_cals, iters")
print("\t",exhaustive(secondF, 0, 1, 0.001))

a, b, fcals, iters = dichotomy(cube, 0, 1, 0.001)
print("\tdichotomy :")
print("\t\ta/f(a): ",a, secondF(a))
print("\t\tb/f(b): ",b, secondF(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)


a, b, fcals, iters = golden_section(secondF, 0, 1, 0.001)
print("\tgolden section :")
print("\t\ta/f(a): ",a, secondF(a))
print("\t\tb/f(b): ",b, secondF(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)


#---------------------------------------------------------------

print("third function:")
print("\texhaustives search :")
print("\t  minimum, f_cals, iters")
print("\t",exhaustive(thirdF, 0.01, 1, 0.001))


a, b, fcals, iters = dichotomy(thirdF, 0.01, 1, 0.001)
print("\tdichotomy :")
print("\t\ta/f(a): ",a, thirdF(a))
print("\t\tb/f(b): ",b, thirdF(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)


a, b, fcals, iters = golden_section(thirdF, 0.01, 1, 0.001)
print("\tgolden section :")
print("\t\ta/f(a): ",a, thirdF(a))
print("\t\tb/f(b): ",b, thirdF(b))
print("\t\tfcals: ", fcals)
print("\t\titers: ",iters)