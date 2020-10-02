import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.optimize import curve_fit


def quadratic(x, a, b, c):
	return a * np.square(x) + b * x + c

def get_rand_points(n):
	points = []
	for i in range(n):
		points.append([np.random.randint(0, 100), np.random.randint(0, 100)])

	return points 



def cross_product_orien(p1, p2, p3):
	return (p2[1] - p1[1]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[1] - p1[1])

def cross_product(p1, p2):
	return p1[0] * p2[1] - p2[0] * p1[1]


def direction(p1, p2, p3):
	return cross_product([p3[0] - p1[0], p3[1] - p1[1]], [p2[0] - p1[0], p2[1] - p1[1]])

def distance(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] + p2[1]) ** 2) ** 0.5

def jarvis_march(points):
	p0 = min(points, key=lambda p: points[0])
	ind = points.index(p0)

	l = ind
	result = []
	result.append(p0)
	k = int(len(points) * (0.2 * len(points)))
	print(len(points))
	print(k)
	while k:
		q = (l + 1) % len(points)
		for i in range(len(points)):
			if i == l:
				continue
			d = direction(points[l], points[i], points[q])
			if d > 0 or (d == 0 and distance(points[i], points[l]) > distance(points[q], points[l])):
				q = i

		l = q
		if l == ind:
			break
		result.append(points[q])
		k -= 1
	return result

n = 200

times = []

for i in range(10, n):
	print(i)
	points = get_rand_points(i)
	start = time.time()
	convex = jarvis_march(points)
	#convex += [convex[0]]
	times.append(time.time() - start)

w, _ = curve_fit(quadratic, range(10, n), times)
y = quadratic(np.array(range(10, n)), w[0], w[1], w[2])

plt.plot(range(10, n), times)
plt.plot(range(10, n), y, color='red')
plt.show()

"""
points = get_rand_points(15)
convex = jarvis_march(points)
convex += [convex[0]]
x, y = (np.array(points)).T
con_x, con_y = (np.array(convex)).T
plt.figure()
plt.plot(con_x, con_y)
plt.scatter(x, y)
plt.show()
"""