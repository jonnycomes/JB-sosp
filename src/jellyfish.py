#written by Ethan Bassingthwaite

#BASIS
'''
For M:
    If m is even the basis consists of all |k> where k is a nonzero even
        integer with |k| ≤ m; 
    If m is odd the basis consists of all |k> where k is an even
        integer with |k| < m.
For N:
    The basis consists of all |k> where k is an odd 
        integer with |k| < n.
'''

#VECTOR SPACES
'''
V = V0 ⊕ V1 denote the superspace with even (resp. odd) part V0 (resp. V1) having
dimension m (resp. n).
I0 for the set of all even integers that correspond to basis vectors of V0.
    When m is even:
        I0 = {k ∈ Z : k is even and 0 < |k| ≤ m}
    When m is odd:
        I0 = {k ∈ Z : k is even and |k| < m}
    For all n:
        I1 = {k ∈ Z : k is odd and |k| < n}.
'''

#OPERATORS
'''
ei,j
    Given i, j ∈ I. Let ei,j : V → V denote the operator that maps |j> → |i> and |k> → 0 whenever k != j.
    ie if |j>, returns |i>, else returns 0.
    We will use ei,j to define the more complicated Ei,j.

Ei,j
    Assumes (j>0) and (i>0) to be treated as their boolean result, TRUE=1.
    Thus we have the super simple definition:
        Ei,j = ei,j - e(-j),(-i) * (-1)^(j + ij + j(j>0) + i(i>0))
    We can find all i,j in I in order to create all the H-type functions we used previously.
'''

#PHI
'''
There should be a sosp(m|n)-map of the form
    φ : V^(⊗m(n+1)) → C

An sosp(m|n)-map means that
    φ(Ei,j · |v>) = 0 for all i, j ∈ I and all |v> ∈ V^(⊗m(n+1))

We should expect there to exist such a φ satisfying:
    1. φ cannot be realized as a linear combination of maps corresponding to (non-jellyfish) Brauer diagrams.
    2. φ satisfies the leg requirement
        a. There are m strands, each of which have thickness n+1.
        b. Permutations with an m-thick strand has no effect on φ.
        c. A crossing of m-thick strands applies a negative sign.

First goal: Find a rule for φ. 
    The rules is given by the determinant in the case n=0

Should use numpy to solve the linear equations (sympy might be better bc we have more equations than variables)

We know that φ(E()) = E(φ()) = 0
So if E() maps it to 0, then we have it vaciously true (as E of a scalar is 0)
Else if E() maps it to a non-zero, then φ on each of the terms must yield a 0
So what we must do is go through and create a system of linear equations using a bunch of bra_kets
For sosp(2|2), the stupid approach will have a lot of checks, but we can start with it as a test
'''

import itertools
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve


class sosp:
    def __init__(self, m, n):
        self.m = m
        self.n = n

        self.I0 = set()
        if m % 2 == 0:
            self.I0 = {k for k in range(-m, m + 1) if k % 2 == 0 and k != 0}
        else:
            self.I0 = {k for k in range(-m, m + 1) if k % 2 == 0}

        self.I1 = {k for k in range(-n, n + 1) if k % 2 == 1}

        self.I = self.I0.union(self.I1)
    
    def __eq__(self, other):
        return self.m == other.m and self.n == other.n
    
    #TODO include this in lin_combo and bra_kets as a value that those objects has in order to determine various things

class lin_combo():
    #Is a linear combination of bra_kets
    #Functionally? It is a list of bra_kets
    def __init__(self, ket_list):
        if ket_list is not None and ket_list != []:
            self.ket_list = [k.standard_form() for k in ket_list]
            #Assume that the ketlist is good
            self.sosp = ket_list[0].sosp
        else:
            self.ket_list = []
            self.sosp = None
            
    def __add__(self, other):
        if isinstance(other, lin_combo):
            #TODO fix this
            return self.reduced_form(self.ket_list+other.ket_list)
        if isinstance(other, bra_ket):
            return lin_combo(self.ket_list + [other]).reduced_form()
        
    def __str__(self):
        return " + ".join(str(val) for val in self.ket_list)
    
    def reduced_form(self):
        #Simplify a lin_combo, no need to simplify the bra_kets as they should be already fixed
        #mostly just reduce and combine same terms and remove 0's
        #assume that all kets are in standard form
        #combine terms
        #we can use a dictionary where the ket part acts as a key in order to add the scalars together
        coefficients = dict()
        for ket in self.ket_list:
            dict_key = tuple(ket.vals)
            coefficients.setdefault(dict_key, 0)
            coefficients[dict_key] += ket.scalar
        
        result = []
        #remove 0's
        for vals, scalar in coefficients.items():
            if scalar != 0:
                result.append(bra_ket(self.sosp, vals, scalar))
        result.sort()
        return lin_combo(result)

    
    

        

class bra_ket():
    #Represents the form |abc....>
    def __init__(self, sosp, vals, scalar = 1):
        self.vals = vals
        self.scalar = scalar
        self.sosp = sosp

    def __mul__(self, other):
        #Applies a tensor multiplication of two bra_kets
        #TODO fix this to include sosp
        return bra_ket(self.vals + other.vals, self.scalar*other.scalar)
    
    def __add__(self, other):
        #Takes the two bra_kets and returns a linear combination
        #TODO include the option of when added to a lin_combo
        #TODO include sosp checking for all functions like this in in this class and lin_combo
        return lin_combo([self, other])
    
    def __str__(self):
        return f"{self.scalar}|{self.vals}>"
    
    def __eq__(self, other):
        #assume that both brakets are in standard form
        #Does not check the scalar value
        return self.vals == other.vals and self.sosp == other.sosp
    
    def __lt__(self, other):
        if len(self.vals) != len(other.vals) or self.sosp != other.sosp:
            print(f"Bra_kets have unequal lengths")
            return len(self.vals) < len(other.vals)
        if self == other:
            return False
        for i in range(len(self.vals)):
            if self.vals[i] < other.vals[i]:
                return True
            if self.vals[i] > other.vals[i]:
                return False
        return False
            

    def standard_form(self):
        #changes the bra_ket into a standardized form
        #Needs to organize both the m-thick strands and the n+1 substrands consistently. 
        #Lets do substrands from least to greatest and then strands by sum of substrands from least to greatest
        #TODO implement a standard form. 
        #TODO handle empty brakets
        '''
            2. φ satisfies the leg requirement
            a. There are m strands, each of which have thickness n+1.
            b. Permutations with an m-thick strand has no effect on φ.
            c. A crossing of m-thick strands applies a negative sign.
        '''     
        if len(self.vals)%(self.sosp.m) != 0:
            print("There exists a non-φ compatible value")
            return self
        #insert sorting algo here in order to make things nice
        #Sort interior of m-stands
        # then we can make the interior of a strand into a braket and then use the inequality definition to sort the strands
        #For now I will not make any changes to the form bc I the changes I would make keep it equal only under the phi
        
        
        return self
    
    def Apply_E(self, i, j):
        #returns a lin_combo derived from applying the Ei,j function to this bra_ket
        #TODO test this on a prior example from my notes to see if the jumping odd numbers applies. 
        #note that if it is a 0, we simply take the end result and set the scalar to 0 as well so that it may be removed
        result = lin_combo([])
        for iter, val in enumerate(self.vals):
            #Determine the odd/even func application
            #TODO double check with Comes
            Oddness = (-1)**(sum(self.vals[:iter])*(i+j))

            #modify the single val and iterate it to a lin_combo
            #bra_kets can be added into 
            result += bra_ket(self.sosp, self.vals[:iter] + [self.E_val(i, j, val)] + self.vals[iter + 1:], self.scalar*self.E_sign(i, j, val)*Oddness).standard_form()

        return result

    def e(self, i, j, val):
        if (j == val):
            return i
        return 0
    
    def E_val(self, i, j, val):
        if self.e(i,j, val) != 0:
            return self.e(i,j, val)
        if self.e(-j,-i, val) != 0:
            return self.e(-j,-i, val)
        return 0

    def E_sign(self, i, j, val):
        #Odd/even functions are applied in Apply_E
        if self.e(i,j, val) != 0:
            return 1
        if self.e(-j,-i, val) != 0:
            return (-1)**(j + i*j + j*(j>0) + i*(i>0) + 1)
        return 0


def phi_solver(m, n):
    #first we need to iterate through all potential values that we could have
    #it is of length m*(n+1)
    length = m*(n+1)
    SOSP = sosp(m,n)

    #Take all possible phi inputs
    #TODO change this into a for loop so that there is no memory error
    all_inputs = list(itertools.product(SOSP.I, repeat=length))

    #Dont want to exclude any inputs thus far
    '''for input in all_inputs:
        if sum(input) != 0:
            all_inputs.remove(input)'''
    #I need a matrix to host all the variables
    #So I will have a 'key' list of all the braket.vals
    #and then I will align their scalar to their position in the key list
    # #For each input, we will have a row, so the sparse matrix is at most a len(all_inputs)^2
    key_list = []
    big_matrix = sparse.lil_matrix((len(all_inputs)*len(SOSP.I)*len(SOSP.I), len(all_inputs)*len(SOSP.I)*len(SOSP.I)), dtype=float)
    print(len(all_inputs))
    line_num = 0
    solution_vector = [0]*len(all_inputs)*len(SOSP.I)*len(SOSP.I)
    
    for input in all_inputs:
        #Apply all possible combinations of E
        for i in SOSP.I:
            for j in SOSP.I:
                #Take and apply the E func
                single_equation = bra_ket(SOSP, list(input)).Apply_E(i, j)

                for term in single_equation.ket_list:
                    if term.vals not in key_list:
                        key_list.append(term.vals)
                    big_matrix[line_num, key_list.index(term.vals)] = term.scalar
                line_num += 1
        print("Progress:", line_num/(len(all_inputs)*len(SOSP.I)*len(SOSP.I)))
    
    big_matrix = big_matrix.tocsr()
    print(big_matrix)
    print(spsolve(big_matrix, solution_vector))

                    

    


###########################################
#
### Test code to see how stuff is working
#
###########################################
print(bra_ket(sosp(2,2), [2, -1,-1, -2, 1, 1]).Apply_E(2, 1))
phi_solver(1, 2)

