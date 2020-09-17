from statistics import median

def quicksort(x):
	if len(x) <= 1:
		return x
	else:
		pivot = median([x[0], x[int(len(x)/2)], x[len(x) - 1]])

	lower = [i for i in x if i < pivot]
	equal = [i for i in x if i == pivot]
	higher = [i for i in x if i > pivot]
	return quicksort(lower) + equal + quicksort(higher)
	