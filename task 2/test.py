import matplotlib.pyplot as plt
import random

a, b = 1, 2

x = []
y = []
for i in range(0, 100):
	l = random.uniform(0, 0.2)
	l = 0
	x.append(i / 100.0)
	y.append(a * i / 100.0 + b )

#x = np.array(x)
#y = np.array(y)	

lr = 0.001
epochs = 1000
error = []

for epoch in range(epochs):
	epoch_cost, cost_a, cost_b = 0, 0, 0

	for i in range(len(x)):
		y_pred = (a * x[i] + b)
		epoch_cost += (y[i] - y_pred)**2

		for j in range(len(x)):
			partial_b = -2 * (y[j] - (a * x[j] + b))
			partial_a = -2 * x[j] * (y[j] - (a * x[j] + b))

			cost_a += partial_a
			cost_b += partial_b
		b -= lr * cost_b
		a -= lr * cost_a

y_preds = []
for i in range(len(x)):
		y_preds.append(a * x[i] + b) 

print(a, b)
plt.plot(x, y, color="red")
plt.plot(x, y_preds)
plt.show()


