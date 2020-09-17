import numpy as np
import random
import matplotlib.pyplot as plt
import linearReg as lr

#a = random.uniform(0, 1)
#b = random.uniform(0, 1)
a = 1
b = 1

def gen_data():
	x = []
	y = []
	n = 100
	for i in range(0, n):
		l = np.random.normal(0.5, 0.01)
		l = 0
		x.append(i / n)
		y.append(a * i / n + b + l)
	return x, y

x, y = gen_data()
x = np.array(x)
y = np.array(y)

w_linear = lr.linear_reg(x, y)
w_rat = lr.rational_reg(x, y)

y_lin = lr.linear_f2(x, w_linear)
y_rat = lr.rational_f2(x, w_rat)

def exhaustive(f, a, b, w, e=0.001):
	n = int((b - a) / e)
	mini = float('inf')
	ind = 1
	for i in range(1, n + 1):
		x_m = a + i * (b - a) / n
		if mini > f(x_m, w):
			mini = f(x_m, w)
			ind = i
	return a + ind * (b - a) / n

def gauss(f, a, b, w, e=0.001):
	return 1

def nelder_1d(f, w, e=0.001):
	ref = 1
	mini = float("inf")
	x_res = 0
	shrink = 0.5
	dilat = 2
	iters = 500
	d = []
	for i in range(3):
		x = random.uniform(0, 1)
		d.append([x, f(x, w)])
	while iters:
		iters -= 1
		d = sorted(d, key=lambda x:x[0])
		xc = 0.5 * (d[0][0] + d[1][0])
		xr = (1+ref) * xc - ref * d[2][0]
		yr = f(xr, w)
		if yr < d[0][1]:
			xe = (1-dilat)*xc + dilat*xr
			ye = f(xe, w)
			if ye < yr:
				d[2][0] = xe
				d[2][1] = ye
			elif yr < ye:
				d[2][0] = xr
				d[2][1] = yr
			elif d[0][1] < yr < d[2][1]:
				xr, d[2][0] = d[2][0], xr
				yr, d[2][1] = d[2][1], yr
				xs = shrink * d[2][0] + (1 - shrink) * xc
				ys = f(xs, w)
				if ys < d[2][1]:
					d[2][0] = xs
					d[2][1] = ys
				else:
					d[1][0] = d[0][0] + (d[1][0] - d[0][0]) / 2.0
					d[2][0] = d[0][0] + (d[2][0] - d[0][0]) / 2.0
					d[1][1] = f(d[1][0], w)
					d[2][1] = f(d[2][0], w)
		c =  1/3.0 * (d[0][0] + d[1][0] + d[2][0])
		if mini > f(c, w):
			mini = f(c, w)
			x_res = c
	return x_res, mini


					


#print(exhaustive(lr.linear_f, 0, 1, w_linear))
print(nelder_1d(lr.rational_f, w_rat))
