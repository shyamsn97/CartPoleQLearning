import numpy as np

a = np.loadtxt("solved_weights.txt")
a = a*7
np.savetxt('test.txt',a)