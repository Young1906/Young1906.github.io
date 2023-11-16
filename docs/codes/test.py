import numpy as np
from matplotlib import pyplot as plt

def I():
    g = lambda x: (1 - x**2)/np.log(x)
    vG = np.vectorize(g)
    x = np.random.uniform(0, 1, 10000)
    return vG(x).mean()

if __name__ == "__main__":
    appx = np.array([I() for _ in range(100)])
    plt.hist(appx, bins=10, color="gray", alpha=.5, label="MC Integration")
    plt.axvline(-np.log(3), color="red", ls="-.", label=r'$-\ln3 \approx {:.3f}$'.format(-np.log(3)))
    plt.legend()
    plt.title(f"Mean: {appx.mean():.3f}; Std: {appx.std():.3f}")
    plt.savefig("../images/mcmc_integral.png")
    plt.show()
