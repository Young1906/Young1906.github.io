---
title: Learning to solve heat equation 
date : 2024-07-22
tags : [learn, ml, pde, pinn]
draft: True 
categories: [
    "Machine Learning",
    "PDE",
    "Learning"
    ]
---

## Heat equations

$$
\begin{aligned}
PDE: & & u_t = \alpha^2 u_{xx} & & 0 < x< 1 & & 0 < t < \infty \\\
BCs: & & \begin{cases}
u(0, t) = 0\\\
u(1, t) = 0
\end{cases} & & 0 < t < \infty \\\
ICs: & & u(x, 0) = \sin(2\pi x) & & 0 \leq x \leq 1
\end{aligned}
$$


## Solving heat equation with variables seperation

Suppose that we can factorize \\(u(x, t) = X(x)T(t)\\), from the PDE we have:

$$
\begin{aligned}
& X(x)T^\prime(t) = \alpha^2 X^{\prime\prime}(x)T(t)\\\
\implies & \frac{T^\prime(t)}{\alpha^2 T(t)} = \frac{X^{\prime\prime}(x)}{X(x)} = \mu \\\
\implies & \begin{cases}
T^\prime(t) - \mu\alpha^2 T(t) = 0 & & (1) \\\ 
X^{\prime\prime}(x) - \mu X(x) = 0 & & (2)
\end{cases}
\end{aligned}
$$

From eq (1), \\(T(t) = Ae^{\mu\alpha^2t}\\). This implies \\(\mu\\) must be negative so that \\(T\\) doesn't go to \\(\infty\\). Let \\(\mu = -\lambda^2\\), so \\(T(t) = Ae^{-\lambda^2\alpha^2t}\\). Replacing into (2), we have:

$$
\begin{aligned}
& X^{\prime\prime}(x) + \lambda^2 X(x) = 0 \\\
\implies & X(x) = B \sin\lambda x + C\cos\lambda x
\end{aligned}
$$

Substitute \\(T(t), X(x)\\) into \\(u(x, t)\\):

$$
u(x, t) = e^{-\lambda^2\alpha^2 t}(A\sin\lambda x + B\cos\lambda x)
$$


Subsititute this into boundary conditions:

$$
\begin{aligned}
& \begin{cases}
u(0, t) = 0 \\\
u(1, t) = 0
\end{cases} \\\
\implies & \begin{cases}
e^{-\lambda^2\alpha^2 t}(A \sin 0 + B \cos 0)= 0 \\\
e^{-\lambda^2\alpha^2 t}(A \sin \lambda + B \cos \lambda) = 0 \\\
\end{cases}\\\
\implies & \begin{cases}
B = 0 \\\
\lambda = n\pi & n = 1, 2, \cdots
\end{cases}
\end{aligned}
$$


So for a given \\(n\\), we have a particular solution for \\(u(x, t)\\):

$$
u_n(x, t) = A_n e^{-n^2\pi^2\alpha^2 t} \sin n\pi x
$$

And the general solution for \\(u(x, t)\\):

$$
u(x, t) = \sum_{n=1}^\infty{A}_n e^{-n^2\pi^2\alpha^2t} \sin n\pi x
$$


Where \\(A_n\\) is given by:

$$
A_n = 2\int_0^1 \phi(x)\sin n\pi x dx
$$

\\(\phi(x) = \sin 2\pi x \\) is the initial condition 

## Finite Difference Method

{{< collapse summary="Finite difference method" >}}
```python
def solve_fdm(nx: int, nt: int, tx: float):
    """
    solving for:
    PDE: u_t = u_xx
    BCs: u(0, t) = u(1, t) = 0
    ICs: u(x, 0) = x - x**2
    """

    x_grid = np.linspace(0, 1, nx)
    dt = tx / nt

    # approximate the result 
    U = np.zeros((nx, nt))

    # IC 
    # ic = lambda x: (x - x ** 2)*100
    ic = lambda x: np.sin(2 * np.pi * x)
    U[:, 0] = np.vectorize(ic)(x_grid)

    ker = 1000 * np.array([1., -2., 1.], dtype=np.float64)/tx**2

    for i in range(1, nt):
        ut = np.convolve(U[:, i - 1], ker, mode="valid")
        U[1:-1,i] = dt * ut + U[1:-1, i-1]

    return U
```
{{< /collapse >}} 



{{< collapse summary="Animation" >}}
```python
def animate_heat():
    # solve heat equation
    U = solve_fdm(100, 10000, 1)
    # Down sampling U
    U = U[:, ::100]

    fig, ax = plt.subplots()
    ln, = ax.plot([], [], lw=1, color="black")

    def init():
        x = np.linspace(0, 1, 100)
        u = np.sin(2 * np.pi * x)

        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)

        ln.set_data(x, u)
        return ln,

    def update(frame):
        x = np.linspace(0, 1, 100)
        u = U[:, frame] 
        ln.set_data(x, u)
        ax.set_title(f"t={frame/100:.2f}s")
        return ln,

    ani = FuncAnimation(fig, update, frames=np.arange(1, 100), init_func=init,
                        blit=True)

    # writer = PillowWriter(fps=60, bitrate=1800)
    # ani.save("heat.gif", writer=writer)
    # plt.close()
    plt.show()
```
{{< /collapse >}}

![img](/images/heat.gif)


## Physics Informed Neural Network
