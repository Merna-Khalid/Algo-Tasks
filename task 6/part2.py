import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import heapq

n = 10
obs = 30

arr = np.full((n,n), 99)
count = obs

while count:
	i = np.random.randint(0, n)
	j = np.random.randint(0, n)
	if arr[i][j] == 99:
		arr[i][j] = 0
		count -= 1

adj = {}


for i in range(n):
	for j in range(n):
		t = []
		if i > 0:
			if arr[i - 1][j] == 99:
				t.append((i - 1, j))
			if j > 0:
				if arr[i - 1][j - 1] == 99:
					t.append((i - 1, j - 1))
				if arr[i][j - 1] == 99:
					t.append((i, j - 1))
			if j < n - 1:
				if arr[i][j + 1] == 99:
					t.append((i, j + 1))
				if arr[i  - 1][j + 1] == 99:
					t.append((i - 1, j + 1))				
		if i < n - 1:
			if arr[i + 1][j] == 99:
				t.append((i + 1, j))
			if j > 0:
				if arr[i + 1][j - 1] == 99:
					t.append((i + 1, j - 1))
			if j < n - 1:
				if arr[i + 1][j + 1] == 99:
					t.append((i + 1, j + 1))
				
		adj[(i, j)] = t

plt.matshow(arr)
plt.show()

class PriorityQueue:
	def __init__(self):
		self.elements = []

	def empty(self):
		return len(self.elements) == 0

	def put(self, item, prior):
		heapq.heappush(self.elements, (prior, item))

	def get(self):
		return heapq.heappop(self.elements)[1]

def distance(a, b):
	(x1, y1) = a
	(x2, y2) = b 
	return abs(x1 - x2) + abs(y1 - y2)
	
def a_star(adj, root, final):
	frontier = PriorityQueue()
	frontier.put(root, 0)
	came_from = {}
	cost_so_far = {}
	came_from[root] = None
	cost_so_far[root] = 0
	while not frontier.empty():
		current = frontier.get()
		if current == final:
			break

		for v in adj[current]:
			new_cost = cost_so_far[current] + 99
			if v not in cost_so_far or new_cost < cost_so_far[v]:
				cost_so_far[v] = new_cost
				priority = new_cost + distance(final, v)
				frontier.put(v, priority)
				came_from[v] = current


	return came_from, cost_so_far

v1 = (0, 0)
v2 = (0, 0)
while True:
	i = np.random.randint(0, n)
	j = np.random.randint(0, n)
	if arr[i][j] == 0:
		continue
	else:
		v1 = (i, j)
		break

while True:
	i = np.random.randint(0, n)
	j = np.random.randint(0, n)
	if arr[i][j] == 0:
		continue
	else:
		v2 = (i, j)
		break
print("v1 : ", v1)
print("v2 : ", v2)
came_from, cost_so_far = a_star(adj, v1, v2)
print("Path : ", came_from)
print("Cost all : ", cost_so_far)
print("cost final", cost_so_far[v2])

t = 0

for k in range(5):
	v1 = (0, 0)
	v2 = (0, 0)
	while True:
		i = np.random.randint(0, n)
		j = np.random.randint(0, n)
		if arr[i][j] == 0:
			continue
		else:
			v1 = (i, j)
			break

	while True:
		i = np.random.randint(0, n)
		j = np.random.randint(0, n)
		if arr[i][j] == 0:
			continue
		else:
			v2 = (i, j)
			break

	start = time.time()
	a_star(adj, v1, v2)
	t += time.time() - start

t /= 5
print("A* 5 times: ", t)
