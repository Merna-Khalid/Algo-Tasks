import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx



n = 100
e = 200

g = nx.Graph()

arr = np.zeros((n,n))

count = int(e / 2) + 1

while count:
	i = np.random.randint(0, n)
	j = np.random.randint(0, n)
	if arr[i][j] == 0:
		g.add_edge(i, j)
		arr[i][j] = 1
		arr[j][i] = 1
		count -= 1

adj = {}

for i in range(n):
	t = []
	for j in range(n):
		if arr[i][j] == 1:
			t.append(j)
	adj[i] = t

for i in range(20):
	for j in range(20):
		print(arr[i][j], end=" ")
	print("")

for i in range(20):
	print(i, " : ", adj[i])

nx.draw(g, with_labels=True)
plt.savefig("filename.png")

print("")
print("")

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

print(connectedComponents(adj))

print("")
print("")


def bfs(graph, root, final):
	if root == final:
		print("Same node")
		return []

	queue = [[root]]
	while queue:
		path = queue.pop(0)
		node = path[-1]
		if node == final:
			return path
		for v in graph[node]:
			new_path = list(path)
			new_path.append(v)
			queue.append(new_path)
	print("no path was found")
	return []

i = np.random.randint(0, n)
j = np.random.randint(0, n)

print("starting vertex : ", i)
print("goal : ", j)
print(bfs(adj, i, j))
