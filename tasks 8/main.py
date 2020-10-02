import eertree as ert 
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np




def add_complexity(x, a, b):
	return a * x * np.log(26) + b

def traverse_odd(x, a, b, c):
	return a * np.square(x) + b * x + c

s = 'rtr'
temp = ('e' * (10 - 3)) + s + ('e' * (10 - 3))
eertree = ert.Eertree()
result = []

for c in temp:
	eertree.add(c)
eertree.get_all_palin(eertree.first_root, [eertree.first_root], [], result) #odd length words
eertree.get_all_palin(eertree.second_root, [eertree.second_root], [], result) #even length words
print("palindromes : ", result)

"""
n = 1000
s = 'rtr'

times = []
for i in range(3, n):
	print(i)
	temp = ('e' * (i - 3)) + s + ('e' * (i - 3))
	time_t = 0
	for j in range(5):
		eertree = ert.Eertree()
		start = time.time()
		for c in temp:
			eertree.add(c)
		#eertree.get_all_palin(eertree.first_root, [eertree.first_root], [], result) #odd length words
		#eertree.get_all_palin(eertree.second_root, [eertree.second_root], [], result) #even length words
		time_t += time.time() - start
	times.append(time_t / 5.0)

w, _ = curve_fit(add_complexity, range(3, n), times)
y = add_complexity(np.array(range(3, n)), w[0], w[1])

plt.plot(range(3, n), times)
plt.plot(range(3, n), y, color='red')
plt.show()


n = 997
s = 'rtr'

times = []
for i in range(3, n):
	print(i)
	temp = ('e' * (i - 3)) + s + ('e' * (i - 3))
	time_t = 0
	for j in range(5):
		eertree = ert.Eertree()
		for c in temp:
			eertree.add(c)
		start = time.time()
		result = []
		eertree.get_all_palin(eertree.first_root, [eertree.first_root], [], result) #Odd length words
		eertree.get_all_palin(eertree.second_root, [eertree.second_root], [], result) #Even length words
		time_t += time.time() - start
	times.append(time_t / 5.0)

w, _ = curve_fit(traverse_odd, range(3, n), times)
y = traverse_odd(np.array(range(3, n)), w[0], w[1], w[2])

plt.plot(range(3, n), times)
plt.plot(range(3, n), y, color='red')
plt.show()

"""
