def merge(x, y):
	r = []
	iter1 = 0
	iter2 = 0

	while iter1 != len(x) and iter2 != len(y):
		if x[iter1] < y[iter2]:
			r.append(x[iter1])
			iter1 += 1
		else:
			r.append(y[iter2])
			iter2 += 1

	while iter1 != len(x):
		r.append(x[iter1])
		iter1 += 1

	while iter2 != len(y):
		r.append(y[iter2])
		iter2 += 1

	return r

def mergeSort(x):
	if len(x) == 1:
		return x
	elif len(x) == 2:
		if x[0] > x[1]:
			x[0], x[1] = x[1], x[0]
		return x
	else:
		part1 = mergeSort(x[:int(len(x)/2)])
		part2 = mergeSort(x[int(len(x)/2):])
		return merge(part1, part2)
