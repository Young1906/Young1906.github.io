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
\begin{equation}
\begin{aligned}
PDE: & & u_t = \alpha^2 u_{xx} & & 0 < x< 1 & & 0 < t < \infty \\\
BCs: & & \begin{cases}
u(0, t) = 0\\\
u(1, t) = 0
\end{cases} & & 0 < t < \infty \\\
ICs: & & u(x, 0) = \sin(2\pi x) & & 0 \leq x \leq 1
\end{aligned}
\end{equation}
$$


## Solving heat equation with variables seperation

Suppose that we can factorize \\(u(x, t) = X(x)T(t)\\), from the PDE we have:

$$
\begin{equation}
\begin{aligned}
& X(x)T^\prime(t) = \alpha^2 X^{\prime\prime}(x)T(t)\\\
\implies & \frac{T^\prime(t)}{\alpha^2 T(t)} = \frac{X^{\prime\prime}(x)}{X(x)} = \mu \\\
\implies & \begin{cases}
T^\prime(t) - \mu\alpha^2 T(t) = 0 & & (2a) \\\ 
X^{\prime\prime}(x) - \mu X(x) = 0 & & (2b)
\end{cases}
\end{aligned}
\end{equation}
$$

From equation (2a), \\(T(t) = Ae^{\mu\alpha^2t}\\). This implies \\(\mu\\) must be negative so that \\(T\\) doesn't go to \\(\infty\\). Let \\(\mu = -\lambda^2\\), so \\(T(t) = Ae^{-\lambda^2\alpha^2t}\\). Replacing into (2), we have:

$$
\begin{equation}
\begin{aligned}
& X^{\prime\prime}(x) + \lambda^2 X(x) = 0 \\\
\implies & X(x) = B \sin\lambda x + C\cos\lambda x
\end{aligned}
\end{equation}
$$

Substitute \\(T(t), X(x)\\) into \\(u(x, t)\\):

$$
\begin{equation}
u(x, t) = e^{-\lambda^2\alpha^2 t}(A\sin\lambda x + B\cos\lambda x)
\end{equation}
$$


Subsititute this into boundary conditions:

$$
\begin{equation}
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
\end{equation}
$$


So for a given \\(n\\), we have a particular solution for \\(u(x, t)\\):

$$
\begin{equation}
u_n(x, t) = A_n e^{-n^2\pi^2\alpha^2 t} \sin n\pi x
\end{equation}
$$

And the general solution for \\(u(x, t)\\):

$$
\begin{equation}
u(x, t) = \sum_{n=1}^\infty{A}_n e^{-n^2\pi^2\alpha^2t} \sin n\pi x
\end{equation}
$$


Where \\(A_n\\) is given by:

$$
\begin{equation}
A_n = 2\int_0^1 \sin 2\pi x\sin n\pi x dx = \begin{cases}
0 \quad \text{if }n \neq 2\\\
1 \quad \text{if }n = 2
\end{cases}
\end{equation}
$$

Finally, we have the solution to the PDE:

$$
\begin{equation}
\blue{u(x, t) = e^{-4\pi^2\alpha^2t}\sin 2\pi x}
\end{equation}
$$


## Finite Difference Method

### Numerical approximation of first and second order derivative

#### First Order Forward Difference 

Consider a Taylor series expansion of \\(\phi(x)\\) about point \\(x_i\\) 

$$
\begin{equation}
\begin{aligned}
    & \phi(x_i + \delta x) = \phi(x_i) 
        + \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} \delta x 
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\delta x^2}{2!}
        + \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\delta x^3}{3!}
        + \cdots \\\
\end{aligned}
\end{equation}
$$

Replace \\(\delta x = \Delta x \ll 1\\) in equation (10):

$$
\begin{aligned}
    & \phi(x_i + \delta x) = \phi(x_i) 
        + \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} \Delta x 
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x^2}{2!}
        + \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^3}{3!}
        + \cdots \\\
    \implies 
    & \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} 
    = \frac{\phi(x_i +\Delta x) - \phi(x_i)}{\Delta x} 
        \red{\underbrace{
        - \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x}{2!}
        - \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^2}{3!} 
        - \cdots
        }_{\text{Truncation error: }\mathcal{O}(\Delta x)}}\\\
    & \blue{\approx \frac{\phi(x_i +\Delta x) - \phi(x_i)}{\Delta x}}
\end{aligned}
$$

> Note that in [this tutorial](http://dma.dima.uniroma1.it/users/lsa_adn/MATERIALE/FDheat.pdf), the truncation error is \\(\mathcal{O}(\Delta x^2)\\). I haven't been able to understand why yet!!!. 


#### First Order Backward Difference 
Replace \\(\delta x = -\Delta x, \Delta x \ll 1\\) in equation (10):

$$
\begin{equation}
\begin{aligned}
    & \phi(x_i + \delta x) = \phi(x_i) 
        - \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} \Delta x 
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x^2}{2!}
        - \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^3}{3!}
        + \cdots \\\
    \implies 
    & \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} 
    = \frac{\phi(x_i) - \phi(x_i - \Delta x)}{\Delta x} 
        \red{\underbrace{
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x}{2!}
        - \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^2}{3!} 
        + \cdots
        }_{\text{Truncation error: }\mathcal{O}(\Delta x)}}\\\
    & \blue{\approx \frac{\phi(x_i) - \phi(x_i - \Delta x)}{\Delta x}}
\end{aligned}
\end{equation}
$$

#### Second Order Central Difference

Replace in equation (10):
- \\(\delta x = \Delta x\\)

$$
\begin{equation}
\begin{aligned}
    & \phi(x_i + \Delta x) = \phi(x_i) 
        + \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} \Delta x 
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x^2}{2!}
        + \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^3}{3!}
        + \cdots 
\end{aligned}
\end{equation}
$$

- \\(\delta x = -\Delta x\\)

$$
\begin{equation}
\begin{aligned}
    & \phi(x_i - \Delta x) = \phi(x_i) 
        - \frac{\partial \phi}{\partial x}\bigg\vert_{x_i} \Delta x 
        + \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x^2}{2!}
        - \frac{\partial^3 \phi}{\partial x^3}\bigg\vert_{x_i} \frac{\Delta x^3}{3!}
        + \cdots 
\end{aligned}
\end{equation}
$$

Adding equation (12) and (13) we have:

$$
\begin{equation}
\begin{aligned}
    & \phi(x_i + \Delta x) + \phi(x_i - \Delta x) =
        2 \phi(x_i)
        + 2 \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} \frac{\Delta x^2}{2!}
        + 2 \frac{\partial^4 \phi}{\partial x^4}\bigg\vert_{x_i} \frac{\Delta x^4}{4!}
        + \cdots\\\
    \implies
    & \frac{\partial^2 \phi}{\partial x^2}\bigg\vert_{x_i} = 
        \frac{\phi(x_i + \Delta x) - 2\phi(x_i) + \phi(x - \Delta x)}{\Delta x^2} 
        \red{\underbrace{
        - 2 \frac{\partial^4 \phi}{\partial x^4}\bigg\vert_{x_i} \frac{\Delta x^2}{4!}
        - \cdots}_{\mathcal{O}(\Delta x^2)}}\\\
    & 
    \blue{
        \approx
        \frac{\phi(x_i + \Delta x) - 2\phi(x_i) + \phi(x - \Delta x)}{\Delta x^2}
    }
\end{aligned}
\end{equation}
$$


### Finite Difference Method for the Heat Equation



Discretize the domain \\(\mathcal{D} = (0, 1) \times (0, T)\\) by constructing a grid
\\(\\{x_i\\}\_{i=1\cdots N} \times \\{t_m\\}\_{m=1\cdots M\}\\). Where: 

- \\(x_i = (i - 1) \Delta x,\quad \Delta x = \frac{1}{N - 1}\\)
- \\(t_m = (m - 1) \Delta t,\quad \Delta t = \frac{T}{M - 1}\\)




#### Forward Time, Centered Space (FTCS)

#### Backward Time, Centered Space (BTCS)


### Implementation 

<!-- {{< collapse summary="Finite difference method" >}}
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
-->

![img](/images/heat.gif)


## Physics Informed Neural Network



# References
- [Finite-Difference Approximations to the Heat Equation](http://dma.dima.uniroma1.it/users/lsa_adn/MATERIALE/FDheat.pdf)
