import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 101)

# 
x = np.cumsum(np.random.randn(101))

plt.plot(t, x)


# filter
for alpha in [ 0.2, 0.5, 0.8]:
    y = np.zeros(101)
    y[0] = x[0]
    for i in range(1,101):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    plt.plot(t, y)

plt.grid(True)
plt.show()