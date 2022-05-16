import numpy as np

# Define the interval class and methods over matrices of intervals

# interval methods
class Interval:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def equals(self, i2):
        if (self.lower == i2.lower) and (self.upper == i2.upper):
            return True
        return False
    
    def display(self):
        return "[" + str(self.lower) + ", " + str(self.upper) + "]"
    
    def __len__(self):
        return 1

# Interval addition. Overloaded to support two intervals, or an interval plus a real number
def int_add(i1, i2):
    assert type(i1) == type(Interval(0,0))

    if type(i2) == type(Interval(0,0)):
        return Interval(i1.lower + i2.lower, i1.upper + i2.upper)
    
    else:
        # assume i2 is a real number
        return Interval(i1.lower + i2, i1.upper + i2)

# Interval multiplication. Overloaded to support two intervals, or an interval times a real number
def int_mult(i1, i2):
    assert type(i1) == type(Interval(0,0))
    
    if type(i2) == type(Interval(0,0)):
        lower = min(i1.lower * i2.lower, i1.lower * i2.upper, i1.upper * i2.lower, i1.upper * i2.upper)
        upper = max(i1.lower * i2.lower, i1.lower * i2.upper, i1.upper * i2.lower, i1.upper * i2.upper)
    
    else:
        # assume i2 is a real number
        lower = min(i1.lower * i2, i1.upper * i2)
        upper = max(i1.lower * i2, i1.upper * i2)

    return Interval(lower, upper)

# Interval subtraction
def int_subtr(i1, i2):
    return int_add(i1, int_mult(i2, -1))

# Transpose a matrix of intervals
def transpose_int_matrix(m):
    if len(m) == 1:
        new_obj = [[Interval(0,0)] for x in range(len(m[0]))]
    
        for k in range(len(m[0])):
            new_obj[k][0] = m[0][k]
        
    elif len(m[0]) == 1:
        new_obj = [[Interval(0,0) for x in range(len(m))]]

        if type(m[0]) == type(Interval(0,0)):
            for k in range(len(m)):
                new_obj[0][k] = m[k]
        else:
            for k in range(len(m)):
                new_obj[0][k] = m[k][0]

    else:
        new_obj = [[Interval(0,0) for x in range(len(m))] for y in range(len(m[0]))]

        for k in range(len(m[0])):
            for j in range(len(m)):
                new_obj[k][j] = m[j][k]
                
    return new_obj

def matr_mult_constant(M, c):
    result = [[Interval(0,0) for x in range(len(M[0]))] for y in range(len(M))]
    for x in range(len(M)):
        for y in range(len(M[x])):
            if c >= 0:
                result[x][y] = Interval(c*M[x][y].lower, c*M[x][y].upper)
            else:
                result[x][y] = Interval(c*M[x][y].upper, c*M[x][y].lower)
    return result


def matr_add(M, N):
    assert(len(M) == len(N))
    assert(len(M[0]) == len(N[0]))

    result = [[Interval(0,0) for x in range(len(M[0]))] for y in range(len(M))]
    for x in range(len(M)):
        for y in range(len(M[x])):
            result[x][y] = int_add(M[x][y], N[x][y])
    return result

# helper function for matrix multiplication over intervals   
def matr_mult_component(row, col):
    assert len(row) == len(col)
    total = Interval(0,0)
    for k in range(len(row)):
        total = int_add(total, int_mult(row[k], col[k]))
        
    return total

# multiply matrices m (d1 x n) and N (n x d2) when both are interval matrices
def matr_mult(m, N):
    
    assert len(m[0]) == len(N)
    
    nT = transpose_int_matrix(N)

    output = [[Interval(0,0) for x in range(len(nT))] for y in range(len(m))]
    for k in range(len(m)):
        for j in range(len(nT)):
            output[k][j] = matr_mult_component(m[k], nT[j])

    return output

# convert a real-valued matrix to an interval-valued matrix
def convert_to_intervals(X, extra_rows):
    # 1-D array (e.g., labels)
    if type(X[0]) == type(1) or type(X[0]) == type(1.1) or type(X[0]) == np.int64:
        result = [Interval(0,0) for k in range(len(X) + extra_rows)]

        for k in range(len(X)):
            result[k] = Interval(X[k],X[k])

        if extra_rows > 0:
            for k in range(extra_rows):
                min_val = min(X)
                max_val = max(X)
                result[k + len(X)] = Interval(min_val, max_val)

        return result
    # 2-D array, i.e., features
    result = [[Interval(0,0) for k in range(len(X[0]))] for j in range(len(X) + extra_rows)]

    for k in range(len(X)):
        for j in range(len(X[k])):
            result[k][j] = Interval(X[k][j], X[k][j])
    
    if extra_rows > 0:
        for j in range(len(X[0])):
            min_val = min(np.transpose(np.matrix(X)).tolist()[j])
            max_val = max(np.transpose(np.matrix(X)).tolist()[j])
            for k in range(extra_rows):
                result[k + len(X)][j] = Interval(min_val, max_val)

    return result

def get_midpoint_interval(val):
    return (val.upper + val.lower)/2

def get_midpoint_matrix(A):
    if (len(A)) == 1 or type(A[0]) == type(Interval(0,0)):
        if len(A) == 1:
            result = np.zeros((len(A[0]), 1))
            for col in range(len(A[0])):
                result[col,0] = get_midpoint_interval(A[0][col])
        else:
            result = np.zeros((len(A), 1))
            for col in range(len(A)):
                result[col,0] = get_midpoint_interval(A[col])
        return result
    result = np.zeros((len(A),len(A[0])))
    for row in range(len(A)):
        for col in range(len(A[0])):
            result[row,col] = get_midpoint_interval(A[row][col])
    return result