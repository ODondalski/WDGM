import numpy as np

# ex 1

ex1 = np.ones(50) * 5
print(ex1)

# ex 2

ex2 = np.arange(1, 26)
ex2_re = ex2.reshape(5, 5)
print(ex2_re)

# ex 3

ex3 = np.arange(10, 51, 2)
print(ex3)

# ex 4

ex4 = np.eye(5) * 8
print(ex4)

# ex 5

ex5 = np.linspace(0, 0.99, 100)
ex5 = ex5.reshape(10, 10)
print(ex5)

# ex 6

ex6 = np.linspace(0, 1, 50)
print(ex6)

# ex 7

ex7 = ex2[11:]
print(ex7)

# ex 8

ex8 = ex2_re[:3, 4:5]

print(ex8)

# ex 9

ex9 = ex2_re[3:5, :]
ex9 = ex9.sum()

print(ex9)


# ex 10

def rand_tens():
    rand_row = np.random.randint(1, 10)
    rand_col = np.random.randint(1, 10)
    rand = np.random.randint(1, 100, (rand_row, rand_col))
    return rand


print(rand_tens())
