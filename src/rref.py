import math

class reducer:
	def __init__(self, col_dim, matrix=None, pivot_cols=None):
		if matrix:
			self.matrix = matrix
		else:
			self.matrix = []
		if pivot_cols:
			self.pivot_cols = pivot_cols
		else:
			self.pivot_cols = []
		self.col_dim = col_dim

	def is_reduced(self):
		return self.col_dim == len(pivot_cols)

	def mult_row(self, row_pos, scalar):
		self.matrix[row_pos] = [r * scalar for r in self.matrix[row_pos]]

	def div_row(self, scalar, row_pos=-1):
		self.matrix[row_pos] = [r // scalar for r in self.matrix[row_pos]]

	def combine_row(self, row, piv_row_pos, scalar, piv_scalar):
		piv_row = self.matrix[piv_row_pos]
		return [a * scalar + b * piv_scalar for (a,b) in zip(row, piv_row)]

	def add_row(self, row):
		'''
		Reduces the given row by the current pivot rows.
		If the newly reduced row is nonzero, it will be added as a new pivot row,
		and used to reduce all the old rows.
		Returns True if a new row is added, False otherwise.
		'''
		new_pivot_col = None

		#Use current pivots to reduce the new row
		for c in range(self.col_dim):
			val = row[c]
			if val != 0:
				if c in self.pivot_cols:
					piv_row_pos = self.pivot_cols.index(c)
					piv_val = self.matrix[piv_row_pos][c]
					row = self.combine_row(row, piv_row_pos, scalar=piv_val, piv_scalar=-val)
				elif new_pivot_col is None:
					new_pivot_col = c
					self.pivot_cols.append(new_pivot_col)

		#If a new pivot is found, add new row and reduce
		if new_pivot_col is not None:
			d = math.gcd(*row)
			self.matrix.append(row)
			self.div_row(d)
			new_piv_val = self.matrix[-1][new_pivot_col]
			for i, old_row in enumerate(self.matrix[:-1]):
				val = old_row[new_pivot_col]
				if val != 0:
					self.matrix[i] = self.combine_row(old_row, piv_row_pos=-1, scalar=new_piv_val, piv_scalar=-val)
					d = math.gcd(*old_row)
					self.div_row(d, row_pos=i)


		return new_pivot_col is not None



