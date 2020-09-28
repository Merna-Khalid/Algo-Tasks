import numpy as np
import random
import matplotlib.pyplot as plt
import linearReg as lr

a = random.uniform(0, 1)
b = random.uniform(0, 1)
#a = 1
#b = 1

def gen_data():
	x = []
	y = []
	n = 100
	for i in range(0, n):
		l = np.random.normal(0, 1)
		x.append(i / n)
		y.append(a * i / n + b + l)
	return x, y

x, y = gen_data()
x = np.array(x)
y = np.array(y)

def D_linear(a, b, x, y):
	return np.sum([(a * x[i] + b - y[i]) ** 2 for i in range(len(x))])

def D_rational(a, b, x, y):
	return np.sum([(a / (1 + b * x[i]) - y[i]) ** 2 for i in range(len(x))])


def exhaust_1d(f, a, b, x, y, xm, e=0.001, first_fixed=True):
	n = int((b - a) / e)
	mini = float('inf')
	ind = 1
	fcals = 0
	iters = 0
	if first_fixed:
		for i in range(1, n + 1):
			x2_m = a + i * (b - a) / n
			d = f(xm, x2_m, x, y)
			fcals += 1
			if mini > d:
				mini = d
				ind = i
			iters += 1
	else:
		for i in range(1, n + 1):
			x1_m = a + i * (b - a) / n
			d = f(x1_m, xm, x, y)
			fcals += 1
			if mini > d:
				mini = d
				ind = i
			iters += 1
	return a + ind * (b - a) / n, fcals, iters

def exhaustive(f, a, b, x, y, e=0.001):
	n = int((b - a) / e)
	fcals = 0
	iters = 0
	mini = float('inf')
	indi = 1
	indj = 1
	for i in range(1, n + 1):
		x1_m = a + i * (b - a) / n
		iters += n
		for j in range(1, n + 1):
			x2_m = a + j * (b - a) / n
			d = f(x1_m, x2_m, x, y)
			fcals += 1
			if mini > d:
				mini = d
				indi = i
				indj = i
	return a + indi * (b - a) / n, a + indj * (b - a) / n, fcals, iters

def gauss(f, a, b, x, y, e=0.001):
	x1_m = random.uniform(0, 1)
	x2_m = random.uniform(0, 1)
	fcals = 0
	iters = 0
	maxIter = 300
	while maxIter:
		prev_x1m = x1_m
		x1_m, ftemps, iters_t = exhaust_1d(f, a, b, x, y, x2_m, first_fixed=False)
		fcals += ftemps
		iters += iters_t
		if abs(x1_m - prev_x1m) < e or f(x1_m, x2_m, x, y) < e:
			break
		prev_x2m = x2_m
		x2_m, ftemps, iters_t = exhaust_1d(f, a, b, x, y, x2_m, first_fixed=False)
		fcals += ftemps
		iters += iters_t
		if abs(x1_m - prev_x1m) < e or f(x1_m, x2_m, x, y) < e:
			break
		maxIter -= 1
	return x1_m, x2_m, fcals, iters



def nelder(f, x, y, e=0.001):
	ref = 1
	mini = float("inf")
	x_res1, x_res2 = 0, 0
	shrink = 0.5
	dilat = 2
	iters = 500
	iters_counter = 0
	fcals = 0
	d = []
	for i in range(3):
		iters
		x1_m = random.uniform(0, 1)
		x2_m = random.uniform(0, 1)
		d.append([x1_m, x2_m, f(x1_m, x2_m, x, y)])
	while iters:
		iters -= 1
		d = sorted(d, key=lambda x:x[2])
		xc1 = 0.5 * (d[0][0] + d[1][0])
		xc2 = 0.5 * (d[0][1] + d[1][1])
		xr1 = (1+ref) * xc1 - ref * d[2][0]
		xr2 = (1+ref) * xc2 - ref * d[2][1]
		xr, xc = 0, 0
		yr = f(xr1, xr2, x, y)
		fcals += 1
		if yr < d[0][2]:
			xe1 = (1-dilat)*xc1 + dilat*xr1
			xe2 = (1-dilat)*xc2 + dilat*xr2
			ye = f(xe1, xe2, x, y)
			fcals += 1
			if ye < yr:
				d[2][0] = xe1
				d[2][1] = xe2
				d[2][2] = ye
			elif yr < ye:
				d[2][0] = xr1
				d[2][1] = xr2
				d[2][2] = yr
			elif d[0][2] < yr < d[1][2]:
				d[2][0] = xr1
				d[2][1] = xr2
			elif d[0][2] < yr < d[2][2]:
				d[2][0], xr1 = xr1, d[2][0]
				d[2][1], xr2 = xr2, d[2][1]
				yr, d[2][1] = d[2][1], yr
				xs1 = shrink * d[2][0] + (1 - shrink) * xc1
				xs2 = shrink * d[2][1] + (1 - shrink) * xc2
				ys = f(xs1, xs2, x, y)
				fcals += 1
				if ys < d[2][1]:
					d[2][0] = xs1
					d[2][1] = xs2
					d[2][2] = ys
				else:
					d[1][0] = d[0][0] + (d[1][0] - d[0][0]) / 2.0
					d[1][1] = d[0][1] + (d[1][1] - d[0][1]) / 2.0
					d[2][0] = d[0][0] + (d[2][0] - d[0][0]) / 2.0
					d[2][1] = d[0][1] + (d[2][1] - d[0][1]) / 2.0
					d[1][2] = f(d[1][0], d[1][1], x, y)
					d[2][2] = f(d[2][0], d[2][1], x, y)
					fcals += 2
		c1 =  1/3.0 * (d[0][0] + d[1][0] + d[2][0])
		c2 =  1/3.0 * (d[0][1] + d[1][1] + d[2][1])
		if mini > f(c1, c2, x, y):
			mini = f(c1, c2, x, y)
			fcals += 1
			x_res1 = c1
			x_res2 = c2
		iters_counter += 1
	return x_res1, x_res2, mini, fcals, iters_counter



# Linear

"""
print("Linear:")
print("\texhaustives search :")
a_, b_, fcals, iters = exhaustive(D_linear, -2, 2, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tprecision : ", D_linear(a_, b_, x, y))

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Exhaustive 2d")
plt.show()
"""
"""
print("Linear:")
print("\tGauss:")
a_, b_, fcals, iters = gauss(D_linear, -1, 1, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tfunction calculations:", fcals)
print("\t\tIterations:", iters)
print("\t\tprecision : ", D_linear(a_, b_, x, y))

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Gauss")
plt.show()
"""

"""

print("Linear:")
print("\tNelder:")
a_, b_, mini, fcals, iters = nelder(D_linear, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tfunction calculations:", fcals)
print("\t\tIterations:", iters)
print("\t\tprecision : ", D_linear(a_, b_, x, y))

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Nelder")
plt.show()


#---------------------------------------------------------------

# rational


print("Rational:")
print("\texhaustives search :")
a_, b_, fcals, iters = exhaustive(D_rational, 0.01, 2, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tprecision : ", D_rational(a_, b_, x, y))

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Exhaustive 2d rational")
plt.show()


print("Rational:")
print("\tGauss:")
a_, b_, fcals, iters = gauss(D_rational, 0.01, 1, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tfunction calculations:", fcals)
print("\t\tIterations:", iters)
print("\t\tprecision : ", D_rational(a_, b_, x, y))

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Gauss rational")
plt.show()


print("Rational:")
print("\tNelder:")
a_, b_, mini, fcals, iters = nelder(D_rational, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tfunction calculations:", fcals)
print("\t\tIterations:", iters)
print("\t\tprecision : ", D_rational(a_, b_, x, y))

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Nelder rational")
plt.show()

"""