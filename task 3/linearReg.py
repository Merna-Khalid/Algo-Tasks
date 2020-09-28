import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


def linear_f1(x, w):
	return np.array([w[0] * i + w[1] for i in x])


# x, y -> numpy arrays

def gradient_desc(x, y, w, f):
	e = y - f(x, w)
	#				a 			b
	return np.array([-2* x * e, -2 * e]).sum(axis=1)


def error(x, y, w):
	return 1/len(x) * np.sum(np.square(linear_f1(x, w) - y))

def linear_reg(x, y, lr=0.001):
	e = 999
	w = [random.uniform(0, 1), random.uniform(0, 1)]
	iters = 0
	iter_max = 300
	fcount, grad_count = 0, 0
	while iters < iter_max and e > 0.0001:
		gr = gradient_desc(x, y, w, linear_f1)
		w -= lr * gr
		iters += 1
		e = error(x, y, w)
		fcount += 1
		grad_count += 1
	return w, e, iters, fcount, grad_count



