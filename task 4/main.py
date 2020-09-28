import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import optimize
from numdifftools import Jacobian, Hessian


def func(x):
	return 1 / (x**2 - 3*x + 2)

def F(args, x, y):
	a, b, c, d = args
	return np.array(np.square(y - (a * x + b) / (x**2 + c*x + d))).sum()


def F2(args, x, y):
	a, b, c, d = args
	return y - ((a * x + b) / (x**2 + c*x + d))


def F3(args, x, y):
	a, b, c, d = args
	return ((a * x + b) / (x**2 + c*x + d))


def error(x, y, w):
	return 1/len(x) * F(w, x, y) 

def gradient(args, x, y):
	return Jacobian(lambda args: F(args, x, y))(args).ravel()

def gen_data():
	x = []
	y = []
	n = 1000
	for i in range(0, n):
		l = np.random.normal(0, 1)
		x.append(3 * i / n)
		t = func(x[-1])
		if t < -100:
			y.append(-100 + l)
		elif t <= 100:
			y.append(t + l)
		else:
			y.append(100 + l)
	return x, y


x, y = gen_data()
x = np.array(x)
y = np.array(y)


res = optimize.minimize(F, [0,1,3,2], method='Nelder-Mead', args=(x, y), options={'maxiter':10, 'fatol':1e-15, 'adaptive':True})

a_, b_, c_, d_ = res.x
e = error(x, y, [a_, b_, c_, d_])
print("\tNelder-mead:")
print("\t\ta / b / c / d: ", a_, " / ", b_, " / ", c_, " / ", d_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Nelder-Mead")
plt.show()

coefs, _, res, msg, _ = optimize.leastsq(F2, [0,1,-3,2], args=(x, y), full_output=True)


a_, b_, c_, d_ = coefs
e = error(x, y, [a_, b_, c_, d_])
print("\tLevenberg Marquardt :")
print("\t\ta / b: ", a_, " / ", b_, " / ", c_, " / ", d_)
print("\t\terror: ", e)
print("\t\tfunction ", res["nfev"])
print(msg)

plt.plot(x, y)
pred = [(a_ * x[i] + b_) / (x[i]**2 + c_*x[i] + d_) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Levenberg Marquardt")
plt.show()


res = optimize.dual_annealing(F, bounds=[[-0.1, 0.1], [0, .1], [-3.1, -2.9], [1.9, 2.1]], args=(x, y), maxiter=1000, x0=[1, 1, 1, 1])

a_, b_, c_, d_ = res.x
e = error(x, y, [a_, b_, c_, d_])
print("\tSimulated Annealing:")
print("\t\ta / b / c / d: ", a_, " / ", b_, " / ", c_, " / ", d_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)

plt.plot(x, y)
pred = [(a_ * x[i] + b_) / (x[i]**2 + c_*x[i] + d_) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Simulated Annealing")
plt.show()


res = optimize.differential_evolution(F,bounds=[[-0.1, 0.1], [0, .1], [-3.1, -2.9], [1.9, 2.1]], args=(x, y), maxiter=1000)

a_, b_, c_, d_ = res.x
e = error(x, y, [a_, b_, c_, d_])
print("\tDifferential evolution:")
print("\t\ta / b / c / d: ", a_, " / ", b_, " / ", c_, " / ", d_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)


plt.plot(x, y)
pred = [(a_ * x[i] + b_) / (x[i]**2 + c_*x[i] + d_) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Differential evolution")
plt.show()


def D(a, b, c, d, x, y):
	return np.sum([((a * x[i] + b) / (x[i]**2 + c*x[i] + d) - y[i]) ** 2 for i in range(len(x))])

def boundary(c1, c2, c3, c4):
	return (-0.1 < c1 < 0.2) and (0.01 < c2 < 1) and (-3.1 < c3 < -2.7) and (1.5 < c4 < 2.1) 

def nelder(f, x, y, e=0.001):
	ref = 1
	mini = float("inf")
	x_res1, x_res2, x_res3, x_res4 = 0, 0.13, -3, 2
	shrink = 0.5
	dilat = 2
	iters = 1000
	iters_counter = 0
	fcals = 0
	d = []
	for i in range(3):
		iters
		
		x1_m = random.uniform(-0.1, 0.2)
		x2_m = random.uniform(0.01, 1)
		x3_m = random.uniform(-3.1, -2.7)
		x4_m = random.uniform(1.5, 2.1)
		
		d.append([x1_m, x2_m, x3_m, x4_m, f(x1_m, x2_m, x3_m, x4_m, x, y)])
	while iters:
		iters -= 1
		d = sorted(d, key=lambda x:x[4])
		xc1 = 0.5 * (d[0][0] + d[1][0])
		xc2 = 0.5 * (d[0][1] + d[1][1])
		xc3 = 0.5 * (d[0][2] + d[1][2])
		xc4 = 0.5 * (d[0][3] + d[1][3])
		xr1 = (1+ref) * xc1 - ref * d[2][0]
		xr2 = (1+ref) * xc2 - ref * d[2][1]
		xr3 = (1+ref) * xc2 - ref * d[2][2]
		xr4 = (1+ref) * xc2 - ref * d[2][3]
		yr = f(xr1, xr2, xr3, xr4, x, y)
		fcals += 1
		if yr < d[0][4]:
			xe1 = (1-dilat)*xc1 + dilat*xr1
			xe2 = (1-dilat)*xc2 + dilat*xr2
			xe3 = (1-dilat)*xc3 + dilat*xr3
			xe4 = (1-dilat)*xc4 + dilat*xr4
			ye = f(xe1, xe2, xe3, xe4, x, y)
			fcals += 1
			if ye < yr:
				d[2][0] = xe1
				d[2][1] = xe2
				d[2][2] = xe3
				d[2][3] = xe4
				d[2][4] = ye
			elif yr < ye:
				d[2][0] = xr1
				d[2][1] = xr2
				d[2][2] = xr3
				d[2][3] = xr4
				d[2][4] = yr
			elif d[0][4] < yr < d[1][4]:
				d[2][0] = xr1
				d[2][1] = xr2
				d[2][2] = xr3
				d[2][3] = xr4
			elif d[0][4] < yr < d[2][4]:
				d[2][0], xr1 = xr1, d[2][0]
				d[2][1], xr2 = xr2, d[2][1]
				d[2][2], xr3 = xr3, d[2][2]
				d[2][3], xr4 = xr4, d[2][3]
				yr, d[2][4] = d[2][4], yr
				xs1 = shrink * d[2][0] + (1 - shrink) * xc1
				xs2 = shrink * d[2][1] + (1 - shrink) * xc2
				xs3 = shrink * d[2][2] + (1 - shrink) * xc3
				xs4 = shrink * d[2][3] + (1 - shrink) * xc4
				ys = f(xs1, xs2, xs3, xs4, x, y)
				fcals += 1
				if ys < d[2][4]:
					d[2][0] = xs1
					d[2][1] = xs2
					d[2][2] = xs3
					d[2][3] = xs4
					d[2][4] = ys
				else:
					d[1][0] = d[0][0] + (d[1][0] - d[0][0]) / 2.0
					d[1][1] = d[0][1] + (d[1][1] - d[0][1]) / 2.0
					d[1][2] = d[0][2] + (d[1][2] - d[0][2]) / 2.0
					d[1][3] = d[0][3] + (d[1][3] - d[0][3]) / 2.0

					d[2][0] = d[0][0] + (d[2][0] - d[0][0]) / 2.0
					d[2][1] = d[0][1] + (d[2][1] - d[0][1]) / 2.0
					d[2][2] = d[0][2] + (d[2][2] - d[0][2]) / 2.0
					d[2][3] = d[0][3] + (d[2][3] - d[0][3]) / 2.0

					d[1][4] = f(d[1][0], d[1][1], d[1][2], d[1][3], x, y)
					d[2][4] = f(d[2][0], d[2][1], d[2][2], d[2][3], x, y)
					fcals += 2
		c1 =  1/3.0 * (d[0][0] + d[1][0] + d[2][0])
		c2 =  1/3.0 * (d[0][1] + d[1][1] + d[2][1])
		c3 =  1/3.0 * (d[0][2] + d[1][2] + d[2][2])
		c4 =  1/3.0 * (d[0][3] + d[1][3] + d[2][3])
		if mini > f(c1, c2, c3, c4, x, y) and boundary(c1, c2, c3, c4):

			mini = f(c1, c2, c3, c4, x, y)
			fcals += 1
			x_res1 = c1
			x_res2 = c2
			x_res3 = c3
			x_res4 = c4
		iters_counter += 1
	return x_res1, x_res2, x_res3, x_res4, mini, fcals, iters_counter


print("\tNelder (by hand):")
a_, b_, c_, d_, mini, fcals, iters = nelder(D, x, y)
print("\t\ta : ", a_)
print("\t\tb : ", b_)
print("\t\tc : ", c_)
print("\t\td : ", d_)
print("\t\tfunction calculations:", fcals)
print("\t\tIterations:", iters)
print("\t\tprecision : ", D(a_, b_, c_, d_, x, y))

plt.plot(x, y)
pred = [(a_ * x[i] + b_) / (x[i]**2 + c_*x[i] + d_) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Nelder (by hand)")
plt.show()
