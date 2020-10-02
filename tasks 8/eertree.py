import numpy as np

class Node:
	def __init__(self):
		self.labels = {}
		self.suffix = None
		for i in range(26):
			self.labels[i] = None
		self.len = 0
		#self.name = ""


class Eertree:
	def __init__(self):
		self.first_root = Node()
		self.first_root.len = -1
		self.first_root.suffix = self.first_root
		#self.first_root.name = "first"
		self.second_root = Node()
		self.second_root.len = 0
		#self.second_root.name = "second"
		self.second_root.suffix = self.first_root
		self.current = self.second_root
		self.s = [0]
		self.nodes = []


	def get_max_suffix(self, ind_node, letter):
		iter_ = ind_node
		l = len(self.s)
		h = iter_.len

		while id(iter_) != id(self.first_root) and self.s[l - h - 1] != letter:
			assert id(iter_) != id(iter_.suffix)
			iter_ = iter_.suffix
			h = iter_.len
		return iter_


	def add(self, letter):
		ind = int(ord(letter) - ord('a'))
		q = self.get_max_suffix(self.current, letter)
		#print("here q", q.name)
		if q.labels[ind] is None:
			temp = Node()
			#temp.name = letter
			self.nodes.append(temp)
			temp.len = q.len + 2
			if temp.len == 1:
				temp.suffix = self.second_root
			else:
				temp.suffix = self.get_max_suffix(q.suffix, letter).labels[ind]
				#print("temp suffix", temp.suffix)
			q.labels[ind] = temp

		self.current = q.labels[ind]
		self.s.append(letter)


	def get_all_palin(self, begin, links, chars, result):
		#print(begin)
		for l in range(26):
			if begin.labels[l] is not None:
				temp = begin.labels[l]
				self.get_all_palin(temp, links + [temp], chars + [chr(l + ord('a'))], result)
		if id(begin) != id(self.first_root) and id(begin) != id(self.second_root):
			out = "".join(chars)
			if id(links[0]) == id(self.second_root):
				result.append(out[::-1] + out)
			else:
				result.append(out[::-1] + out[1:])



