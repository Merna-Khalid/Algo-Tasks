import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time


n = 100
e = 500

g = nx.Graph()

arr = np.zeros((n,n))

count = int(e / 2)



while count:
	i = np.random.randint(0, n)
	j = np.random.randint(0, n)
	if arr[i][j] == 0:
		g.add_edge(i, j)
		arr[i][j] = np.random.randint(4, 100)
		arr[j][i] = arr[i][j]
		count -= 1


adj = {}

for i in range(n):
	t = []
	for j in range(n):
		if arr[i][j] != 0:
			t.append(j)
	adj[i] = t

for i in range(10):
	print(i, " : ", adj[i])

nx.draw(g, with_labels=True)
plt.savefig("filename.png")


def dijkstra(adj, graph, root):
	visited = {root: 0}
	path = {}

	nodes = set(range(n))
	while nodes:
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] < visited[min_node]:
					min_node = node

		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in adj[min_node]:
			weight = current_weight + graph[min_node][edge]
			if edge not in visited or weight < visited[edge]:
				visited[edge] = weight
				path[edge] = min_node
				
	return path, visited

def dfs(adj, temp, v, visited):
	visited[v] = True

	temp.append(v)

	for i in adj[v]:
		if visited[i] == False:
			temp = dfs(adj, temp, i, visited)
	return temp

# adj list
def connectedComponents(graph):
	visited = [False] * len(graph)
	connected = []

	for v in range(len(graph)):
		if visited[v] == False:
			temp = []
			connected.append(dfs(adj, temp, v, visited))
	return connected

print("The connected Components ", connectedComponents(adj))

print("")
print("Dijkstra")
print("")

i = np.random.randint(0, n)

print("i : ", i)
path, visited = dijkstra(adj, arr, i)
print("shortest ", visited)

def bellman_ford(adj, graph, root):
	distances = {}
	for v in range(n):
		distances[v] = 0 if v == root else 999999
	for _ in range(n):
		for i in range(n):
			for j in adj[i]:
				#print(i, j, end= " ")
				if distances[j] > distances[i] + graph[i][j]:
					distances[j] = distances[i] + graph[i][j]
			#print("")

	return distances

print("")
print("Bellman ford")
print("")


print("i : ", i)
print("shortest ", bellman_ford(adj, arr, i))


print("")
print("")

print("Time for dijkstra : ")
t = 0
for i in range(10):
	start = time.time()
	dijkstra(adj, arr, i)
	t += time.time() - start
t /= 10
print("time : ", t)


print("Time for bellman ford : ")
t = 0
for i in range(10):
	start = time.time()
	bellman_ford(adj, arr, i)
	t += time.time() - start
t /= 10
print("time : ", t)


