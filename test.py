import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(2,3,4)
file = open('points')
lines = file.readlines()
i = 0
for line in lines:
    if i > 400:
        break
    ax.scatter(*[float(f) for f in line.split(' ')])
    i = i + 1
plt.show()
