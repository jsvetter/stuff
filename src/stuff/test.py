# %%
import numpy as np
import matplotlib.pyplot as plt

print("This is a test file.")
print(np.arange(10))

# %%
with plt.rc_context(fname="../../matplotlibrc"):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title("Sine Wave")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
