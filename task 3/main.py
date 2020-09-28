import numpy as np
import random
import matplotlib.pyplot as plt
import linearReg as lr
import RatReg as rr
from scipy import optimize
from numdifftools import Jacobian, Hessian
from task2_partTwo import exhaustive, D_linear, D_rational, gauss, nelder

a = random.uniform(0, 1)
b = random.uniform(0, 1)


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


# for scipy functs linear

def linear(args, x, y):
	a, b = args
	return (np.square(y - (a * x + b))).sum()

def linear2(args, x, y):
	a, b = args
	return y - (a * x + b)

def gradient_lin(args, x, y):
	return Jacobian(lambda args: linear(args, x, y))(args).ravel()

#----------------------------------------------------

# for scipy functs rat

def rational(args, x, y):
	a, b = args
	return (np.square(y - a / (1 + b * x))).sum()

def rational2(args, x, y):
	a, b = args
	return y - a / (1 + b * x)

def gradient_rat(args, x, y):
	return Jacobian(lambda args: rational(args, x, y))(args).ravel()
#----------------------------------------------------



#----------------------------------------------------

# linear function
print(a, b)
print("Linear function : ")

# gradient descent
w, e, iters, fcount, grad_count = lr.linear_reg(x, y)

a_, b_ = w

print("\tGradient descent :")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", iters)
print("\t\tfunction: ", fcount)
print("\t\tgradient: ", grad_count)

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Gradient descent")
plt.show()


res = optimize.minimize(linear, [1,1], method='CG', args=(x, y))

a_, b_ = res.x
e = lr.error(x, y, [a_, b_])
print("\tConjugate Gradient descent :")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)
print("\t\tgradient: ", res.njev)

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Conjugate Gradient descent")
plt.show()

res = optimize.minimize(linear, [1,1], method='Newton-CG', args=(x, y), jac=gradient_lin)

a_, b_ = res.x
e = lr.error(x, y, [a_, b_])
print("\tNewton's descent :")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)
print("\t\tgradient: ", res.njev)

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Newton's descent")
plt.show()

coefs, _, res, msg, _ = optimize.leastsq(linear2, [1, 1], args=(x, y), full_output=True)


a_, b_ = coefs
e = lr.error(x, y, [a_, b_])
print("\tLevenberg Marquardt :")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
#print("\t\titerations: ", res["nit"])
print("\t\tfunction ", res["nfev"])
print(msg)
#print("\t\tgradient: ", res["njac"])

plt.plot(x, y)
pred = [(a_ * x[i] + b_ ) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Levenberg Marquardt")
plt.show()


# ////////////////	Task 2  ///////////////////////////////

# Linear

print("")
print("")

print("Linear Task 2:")

print("")

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

print("Linear Task 2:")
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

print("Linear Task 2:")
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

#----------------------------------------------------

#----------------------------------------------------
print("")
print("")

print("Rational function : ")

# Rational function
w, e, iters, fcount, grad_count = rr.rat_reg(x, y)

a_, b_ = w

print("\tGradient descent (rational):")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", iters)
print("\t\tfunction: ", fcount)
print("\t\tgradient: ", grad_count)

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Gradient descent Rational (rational)")
plt.show()


res = optimize.minimize(rational, [1,1], method='CG', args=(x, y))

a_, b_ = res.x
e = rr.error(x, y, [a_, b_])
print("\tConjugate Gradient descent (rational):")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)
print("\t\tgradient: ", res.njev)

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Conjugate Gradient descent Rational (rational)")
plt.show()

res = optimize.minimize(rational, [1,1], method='Newton-CG', args=(x, y), jac=gradient_rat)

a_, b_ = res.x
e = rr.error(x, y, [a_, b_])
print("\tNewton's descent (rational):")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
print("\t\titerations: ", res.nit)
print("\t\tfunction ", res.nfev)
print("\t\tgradient: ", res.njev)

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Newton's descent (rational)")
plt.show()

coefs, _, res, msg, _ = optimize.leastsq(rational2, [1, 1], args=(x, y), full_output=True)


a_, b_ = coefs
e = lr.error(x, y, [a_, b_])
print("\tLevenberg Marquardt (rational):")
print("\t\ta / b: ", a_, " / ", b_)
print("\t\terror: ", e)
#print("\t\titerations: ", res["nit"])
print("\t\tfunction ", res["nfev"])
print(msg)
#print("\t\tgradient: ", res["njac"])

plt.plot(x, y)
pred = [a_ / (1 + b_ * x[i]) for i in range(len(x))]
plt.plot(x, pred, color="red")
plt.title("Levenberg Marquardt (rational)")
plt.show()


#----------------------------------------------------

# ////////////////	Task 2  ///////////////////////////////

print("")
print("")
print("Rational Task 2:")
print("")

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