# 1. Basics
# To use NumPy, you always start by importing it under the standard alias np.

from random import Random
import numpy as np

# Creating a basic 1D array
arr = np.array([1, 2, 3])
print(arr)  # Output: [1 2 3]

# 2. Array Creation
# NumPy offers quick ways to initialize arrays without writing manual loops.

# np.zeros(n) and np.ones(n)
zeros_arr = np.zeros(4)   # Output: [0. 0. 0. 0.]
ones_arr = np.ones((2, 3)) # Output a 2x3 matrix of 1s

# np.arange(start, stop, step) -> Exclusive of 'stop'
range_arr = np.arange(1, 10, 2) 
print(range_arr)  # Output: [1 3 5 7 9]

# np.linspace(start, stop, num) -> Inclusive of 'stop', generates evenly spaced numbers
lines_arr = np.linspace(0, 1, 5)   
print(lines_arr)  # Output: [0.   0.25 0.5  0.75 1.  ]

# 3. Shape & Dimension
# These attributes help you inspect the structure of your data.

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.ndim)  # Output: 2 (It's a 2D array)
print(matrix.shape) # Output: (2, 3) (2 rows, 3 columns)

# 4. Reshaping
# Changing the layout of an array without changing its actual data.

arr = np.arange(1, 7) # [1, 2, 3, 4, 5, 6]
# arr.reshape(rows, cols)
grid = arr.reshape(2, 3)
print(grid)
# Output:
# [[1 2 3]
#  [4 5 6]]

# arr.flatten() -> Turns it back into a 1D array
print(grid.flatten()) # Output: [1 2 3 4 5 6]
# 5. Indexing & Slicing
# Remember: NumPy slicing follows the standard [start:end] format, where end is exclusive. For 2D arrays, use [row, column].

arr = np.array([10, 20, 30, 40, 50])
print(arr[1])     # Output: 20
print(arr[1:4])   # Output: [20 30 40]

# 2D Slicing
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix[0, 1])   # Row 0, Col 1 -> Output: 2
print(matrix[:, 1:3]) # All rows, columns 1 and 2 -> Output: [[2 3], [5 6]]

# 6. Operations
# Element-wise operations are incredibly fast in NumPy compared to Python lists.

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise math
print(a + b)  # Output: [5 7 9]
print(a * b)  # Output: [4 10 18]
print(a ** 2) # Output: [1 4 9]

# np.dot(a, b) -> Matrix / Vector multiplication
# (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
print(np.dot(a, b)) # Output: 32


# 7. Statistics
# Quick functions to summarize your data.


arr = np.array([1, 2, 3, 4, 5])

print(np.mean(arr)) # Output: 3.0
print(np.sum(arr))  # Output: 15
print(np.max(arr))  # Output: 5
print(np.min(arr))  # Output: 1

# 8. Axis
# When dealing with matrices, axis=0 means the operation runs down the columns, and axis=1 means it runs across the rows.

matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])

# Column-wise (vertical)
print(np.sum(matrix, axis=0)) # Output: [5 7 9]
# Row-wise (horizontal)
print(np.sum(matrix, axis=1)) # Output: [6 15]


# 9. Broadcasting
#  Broadcasting allows NumPy to perform operations on arrays of different shapes by "stretching" the smaller array.

arr = np.array([1, 2, 3])

# Broadcasting a scalar (arr + number)
print(arr + 10) # Output: [11 12 13]

# Broadcasting with different shapes
matrix = np.array([[10, 20, 30], 
                   [40, 50, 60]])
# arr is automatically broadcasted to match the rows of the matrix
print(matrix + arr) 
# Output:
# [[11 22 33]
#  [41 52 63]]


# 10. Random
# Essential for data generation, splitting datasets, and initializing weights.

# Setting a seed ensures the "random" numbers are the same every time you run it
np.random.seed(42)

# np.random.rand(n) -> uniform distribution between 0 and 1
print(np.random.rand(3)) 

# np.random.randint(low, high, size)
random_ints = np.random.randint(1, 100, size=5)
print(random_ints) # Output: 5 random integers between 1 and 99


# 11. Filtering (Boolean Masking)
# Instead of writing an if statement for every element, you create a "mask" of Booleans.

arr = np.array([5, 12, 3, 18, 7])
# Single condition
print(arr[arr > 10]) # Output: [12 18]

# Multiple conditions -> Use & (AND), | (OR). Remember brackets!
cond1 = arr > 5
cond2 = arr < 15
print(arr[cond1 & cond2]) # Output: [12 7]


# 12. np.where
# Think of np.where(condition, true_value, false_value) as an element-wise ternary operator (if-else).

scores = np.array([45, 80, 55, 95])

# If score >= 60, 'Pass', else 'Fail'
results = np.where(scores >= 60, 'Pass', 'Fail')
print(results) # Output: ['Fail' 'Pass' 'Fail' 'Pass']


# 13. Useful Functions
# Handy utilities for managing arrays.

a = np.array([3, 1, 2, 2, 1])
b = np.array([4, 5])

print(np.unique(a))      # Output: [1 2 3] (Removes duplicates)
print(np.sort(a))        # Output: [1 1 2 2 3]

# np.concatenate((a, b)) -> Glues arrays together
print(np.concatenate((a, b))) # Output: [3 1 2 2 1 4 5]


# 14. Data Handling
# While Pandas is often used for CSVs, NumPy can quickly load numeric data.

# Imagine a file named 'data.csv' containing:
# 1.0, 2.0, 3.0
# 4.0, 5.0, 6.0

# data = np.loadtxt('data.csv', delimiter=',')
# 15. Normalization (Min-Max Scaling)
# Scales your data so all values fall strictly between 0 and 1.


x = np.array([10, 20, 30, 40, 50])

x_min = np.min(x)
x_max = np.max(x)

normalized_x = (x - x_min) / (x_max - x_min)
print(normalized_x) # Output: [0.   0.25 0.5  0.75 1.  ]


# 16. Standardization (Z-score Normalization)
# Centers data around a mean of 0 with a standard deviation of 1.


x = np.array([10, 20, 30, 40, 50])

mean = np.mean(x)
std = np.std(x)

standardized_x = (x - mean) / std
print(standardized_x) # Output: [-1.414 -0.707  0.     0.707  1.414]