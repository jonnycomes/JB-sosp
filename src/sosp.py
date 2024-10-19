import itertools
import random
import numpy as np
from rref import reducer


class Basis:
	"""Basis for the natural sosp(m|n)-module"""
	def __init__(self, m, n):
		if n % 2 != 0:
			raise Exception('sosp(m|n) requires an even n.')
		self.odd_states = list(range(1-n, 1+n, 2))
		if m % 2 == 0:
			self.even_states = list(range(-m, m+1, 2))
			self.even_states.remove(0)
		else:
			self.even_states = list(range(1-m, m, 2))
		self.states = self.even_states + self.odd_states


class SignedSuperList(list):
	"""A list with even and odd elements along with a sign of 1, -1, or 0"""
	def __init__(self, lst=[]):
		super().__init__(lst)
		self.sign = 1

	def __repr__(self):
		return f'{self.sign} ' + super().__repr__() 

	def __mod__(self, N):
		return sum([elt % N for elt in self]) % N

	def is_balanced(self):
		for i in self:
			if self.count(i) != self.count(-i):
				return False
		return True

	def swap(self, i, j):
		if self[i] % 2 and self[j] % 2:
			self.sign *= -1
		self[i], self[j] = self[j], self[i]

	def sort(self, skew=False):
		'''bubble sort using the (super)swap'''
		for i in range(len(self)):
			for j in range(len(self) - i - 1):
				if self[j] > self[j+1]:
					self.swap(j, j+1)
					if skew:
						self.sign *= -1

	def normal_form(self, m, n):
		if len(self) != m*(n+1):
			raise Exception()
		zero = SignedSuperList()
		zero.sign = 0
		b = Basis(m, n)

		if not self.is_balanced():
			return zero
		slices = SignedSuperList([SignedSuperList(self[i:i+n+1]) for i in range(0, len(self), n+1)])
		for slc in slices:
			for i in b.odd_states:
				if slc.count(i) > 1:
					return zero
			slc.sort()
			slices.sign *= slc.sign
		slices.sort(skew=True)

		nf = SignedSuperList([i for slc in slices for i in slc])
		nf.sign *= self.sign * slices.sign

		return nf

def E(j, k, sslist):
	'''
	Applies the operator E_{i,j} to a SignedSuperList.
	The terms of the resulting sum are stored in a list of SignedSuperLists. 
	Returns the resulting list of SignedSuperLists.
	'''
	out_sslists = []
	sgn = 1
	for i, state in enumerate(sslist):
		if state == k:
			out = SignedSuperList(sslist[:i] + [j] + sslist[i+1:])
			out.sign = sslist.sign * sgn
			out_sslists.append(out)
		if state == -j:
			out = SignedSuperList(sslist[:i] + [-k] + sslist[i+1:])
			out.sign = sslist.sign * (-1)**(1 + k + k*j + k*(k>0) + j*(j>0)) * sgn
			out_sslists.append(out)
		if (j + k) % 2 and state % 2:
			sgn *= -1
	return out_sslists

def normal_forms(m, n):
	nfs = []
	basis = Basis(m, n)

	blocks = []
	for odd_size in range(n+1):
		even_size = n+1 - odd_size
		for odd_part in itertools.combinations(basis.odd_states, odd_size):
			for even_part in itertools.product(basis.even_states, repeat=even_size):
				blocks.append(list(even_part) + list(odd_part))

	nfs = []
	for bs in itertools.product(blocks, repeat=m):
		bs = list(bs)
		t = []
		for b in bs:
			t += b
		ssl = SignedSuperList(t)
		nf = ssl.normal_form(m, n)
		lst = list(nf)
		if ssl.is_balanced() and lst not in nfs and lst: 
			nfs.append(lst)



	# for t in itertools.product(basis.states, repeat=m*(n+1)):
	# 		print(len(nfs), (m+n)**(m*(n+1)) - counter)
	# 	ssl = SignedSuperList(t)
	# 	nf = ssl.normal_form(m, n)
	# 	lst = list(nf)
	# 	if ssl.is_balanced() and lst not in nfs and lst: 
	# 		nfs.append(lst)

	return sorted(nfs)

def generate_aug_matrix(m, n):
	nfs = normal_forms(m, n)
	states = Basis(m, n).states
	odds = Basis(m, n).odd_states
	evens = Basis(m, n).even_states

	#Fix one row, generalizing the determinant 
	row = [0 for _ in range(len(nfs))] + [1]
	fix = [[i] + odds for i in evens]
	fix = [elt for f in fix for elt in sorted(f)]
	col = nfs.index(fix)
	row[col] = 1
	mtx = [row]

	for t in itertools.product(states, repeat=m*(n+1)):
		ssl = SignedSuperList(t)
		for j in states:
			for k in states:
				terms = E(j, k, ssl)
				row = [0 for _ in range(len(nfs))] + [0]
				for term in terms:
					nt = term.normal_form(m, n)
					if list(nt):
						col = nfs.index(list(nt))
						row[col] += nt.sign
				if row != [0 for _ in range(len(nfs))] + [0]:
					mtx.append(row)

	return mtx

def compute_jelly_action(m, n, countdown=True):
	bad_nfs = []
	nfs = normal_forms(m, n)
	basis = Basis(m, n)
	states = basis.states
	odds = basis.odd_states
	evens = basis.even_states

	fix = [[i] + odds for i in evens]
	fix = [elt for f in fix for elt in sorted(f)]

	columns = len(nfs)

	REF = reducer(col_dim=columns)

	for column, nn in enumerate(nfs):
		#Fix one row to generalize the determinant
		if nn == fix:
			row = [0 for _ in range(columns)] + [1]
			row[column] = 1
			REF.add_row(row)
			if countdown:
				print(columns - len(REF.pivot_cols))
		else:
			done = column in REF.pivot_cols
			counter = 0
			while not done and counter < 15:
				counter += 1
				nf = nn[:]
				i = random.randrange(m*(n+1))
				j = nf[i]
				k = random.choice(states)
				nf[i] = k 
				ssl = SignedSuperList(nf)
				terms = E(j, k, ssl)
				row = [0 for _ in range(columns)] + [0]
				for term in terms:
					nt = term.normal_form(m, n)
					if list(nt):
						col = nfs.index(list(nt))
						row[col] = int(row[col] + nt.sign)
				if row != [0 for _ in range(columns)] + [0]:
					done = REF.add_row(row) and column in REF.pivot_cols
					if countdown:
						print(columns - len(REF.pivot_cols))
			if not done:
				for i in range(m*(n+1)):
					for k in states:
						nf = nn[:]
						j = nf[i]
						nf[i] = k 
						ssl = SignedSuperList(nf)
						terms = E(j, k, ssl)
						row = [0 for _ in range(columns)] + [0]
						for term in terms:
							nt = term.normal_form(m, n)
							if list(nt):
								col = nfs.index(list(nt))
								row[col] = int(row[col] + nt.sign)
						if row != [0 for _ in range(columns)] + [0]:
							done = REF.add_row(row)
							if done:
								break
							if countdown:
								print(columns - len(REF.pivot_cols))
						
						ssl = SignedSuperList(nf)
						terms = E(-k, -j, ssl)
						row = [0 for _ in range(columns)] + [0]
						for term in terms:
							nt = term.normal_form(m, n)
							if list(nt):
								col = nfs.index(list(nt))
								row[col] = int(row[col] + nt.sign)
						if row != [0 for _ in range(columns)] + [0]:
							done = REF.add_row(row)
							if done:
								break
							if countdown:
								print(columns - len(REF.pivot_cols))
					else:
						continue
					break
			if not done:
				bad_nfs.append(nn)


	return REF, nfs, bad_nfs



def generate_rref(m, n, countdown=True):
	nfs = normal_forms(m, n)
	basis = Basis(m, n)
	states = basis.states
	odds = basis.odd_states
	evens = basis.even_states

	#Fix one row, generalizing the determinant 
	# row = [0 for _ in range(len(nfs))] + [1]
	fix = [[i] + odds for i in evens]
	fix = [elt for f in fix for elt in sorted(f)]
	# fix_col = nfs.index(fix)
	# row[fix_col] = 1
	# mtx = [row]

	columns = len(nfs)
	mtx = []
	pivot_positions = []

	for col, nn in enumerate(nfs):
		nf = nn[:]
		#Fix one row to generalize the determinant
		if nf == fix:
			row = [0 for _ in range(columns)] + [1]
			row[col] = 1
			mtx.append(row)
			pivot_positions.append(col)
			if countdown:
				print(columns - len(pivot_positions))
		else:
			done = False
			while not done:
				i = random.randrange(m*(n+1))
				j = nf[i]
				k = random.choice(states)
				nf[i] = k 
				ssl = SignedSuperList(nf)
				terms = E(j, k, ssl)
				row = [0 for _ in range(columns)] + [0]
				for term in terms:
					nt = term.normal_form(m, n)
					if list(nt):
						col = nfs.index(list(nt))
						row[col] += nt.sign
				if row != [0 for _ in range(columns)] + [0]:
					new_mtx = mtx + [row]
					rr = rref(np.array(new_mtx))
					if len(rr[1]) > len(pivot_positions):
						pivot_positions = [p[0] for p in rr[1]]
						done = True
						mtx = rr[0].tolist()
						if countdown:
							print(columns - len(pivot_positions))



	# done = False
	# pivots = 0
	# columns = len(nfs)
	# while not done:
	# 	t = random.choices(states, k=m*(n+1))
	# 	ssl = SignedSuperList(t)
	# 	j = random.choice(states)
	# 	k = random.choice(states)
	# 	terms = E(j, k, ssl)
	# 	row = [0 for _ in range(columns)] + [0]
	# 	for term in terms:
	# 		nt = term.normal_form(m, n)
	# 		if list(nt):
	# 			col = nfs.index(list(nt))
	# 			row[col] += nt.sign
	# 	if row != [0 for _ in range(columns)] + [0]:
	# 		new_mtx = mtx + [row]
	# 		rr = rref(np.array(new_mtx))
	# 		if len(rr[1]) > pivots:
	# 			pivots = len(rr[1])
	# 			if pivots == columns:
	# 				done = True
	# 			mtx = rr[0].tolist()
	# 		if countdown:
	# 			print(columns - pivots)

	return mtx, nfs
		



if __name__ == '__main__':

	m = 2
	n = 2
	
	R, nfs, bad_nfs = compute_jelly_action(m, n)

	mtx = R.matrix
	pivot_cols = R.pivot_cols

	coef = {}

	for i, row in enumerate(mtx):
		last = row[-1]
		if last != 0:
			pos = pivot_cols[i]
			val = round(last/row[pos], 5)
			state = nfs[pos]
			if val not in coef:
				coef[val] = []
			coef[val].append(state)

	with open('../doc/output.txt', 'w') as outfile:
		outfile.write(str(dict(sorted(coef.items()))))

	print(bad_nfs)






