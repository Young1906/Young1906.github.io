---
title: Numerical Integrations 
date : 2024-09-26
tags : [learn]
draft: false 
categories: [
    "Machine Learning",
    "PDE",
    "Numerical methods",
    ]
comments: true 
cover:
    image: "/images/ode_fig1.png"
---

> EDITTING


## Ordinary Differential Equation (ODE) Initial Value Problem

A **differential equation** is differential equation is a relationship between function \\(f(x)\\), its independent variable \\(x\\), and any number of its derivative. An **ODE** is a differential equation where the independent variable and its derivatives are in one dimension.

$$
\begin{equation}
F(x, f(x), f^{(1)}(x), f^{(2)}, \cdots f^{(n-1)}(x)) = f^{(n)}(x) 
\end{equation}
$$

Where \\(f^{(i)}\\) is the \\(i^{th}\\) order derivative of \\(f\\). **Initial value** is a set of known value at \\(x = 0\\), namely \\(f(0), f^{(1)}(0), f^{(2)}, \cdots f^{(n-1)}(0)\\). Coupled with equation 1, the problem is known as the ODE Initial Value Problem.

An example is the dampen harmonic occillator with the setup as following diagram:

![fig1](/images/ode_fig1.png)


The motion of mass \\(m\\) is gorverned by set of equations:

$$
\begin{equation}
    \begin{aligned}
        ODE \quad & F(t, x, \dot{x}) = -\frac{c \dot{(x)} + kx}{m} = \ddot{x}  \\\
        IV \quad & x(0) = A\\\
        & \dot{x}(0) = 0
    \end{aligned}
\end{equation}
$$

### Reduction or Order

Denote \\(S(x)\\) to be a state of equation (1):

$$
\begin{equation}
\begin{aligned}
    & S(x) = \begin{bmatrix}
        f(x)                \\\
        f^{(1)}(x)          \\\
        \vdots              \\\
        f^{(n-1)}(x)        \\\
    \end{bmatrix} \\\
\end{aligned}
\end{equation}
$$

So equation 1 can be rewriten as 

$$
\begin{equation}
\begin{aligned}
    F(t, S(t)) = f^{(n)}(t)
\end{aligned}
\end{equation}
$$


Taking derivative of \\(S\\)

$$
\begin{equation}
\begin{aligned}
    & 
        \frac{dS}{dt}  = \begin{bmatrix}
        f^{(1)}(x)          \\\
        f^{(2)}(x)          \\\
        \vdots              \\\
        f^{(n)}(x)          \\\
    \end{bmatrix} = \begin{bmatrix}
        S_2(x)          \\\
        S_3(x)          \\\
        \vdots              \\\
        F\big(x, S(t))
    \end{bmatrix} & \text{(Equation 3)} \\\
    & = \mathcal{F}(t, S(t))
\end{aligned}
\end{equation}
$$

Where \\(S_i(x)\\) is the \\(i^{th}\\) entry of \\(S(x)\\). The \\(n^{th}\\) order ODE is turned into \\(n\\) coupled ODEs, where \\(\mathcal{F}\\) is a function that assemble the correct state vector.

Back to the dampend harmonic occilliator, we can denote the state vector \\(S(t)\\) as:

$$
\begin{equation}
\begin{aligned}
& S(t) = \begin{bmatrix}
    x(t)\\\
    \dot{x}(t)
\end{bmatrix} \\\
\implies &
    \frac{dS}{dt} = \begin{bmatrix}
        \dot{x}(t)\\\
        \ddot{x}(t)
    \end{bmatrix} \\\
& = \begin{bmatrix}
        S_2(t)\\\
        F(t, x, \dot{x})
    \end{bmatrix} & \text{(ODE in 2)} \\\
& = \begin{bmatrix}
    0 & 1 \\\
    -k/m & -c/m
\end{bmatrix} \begin{bmatrix}
    x(t)\\\
    \dot{x}(t) 
\end{bmatrix} \\\
& = \begin{bmatrix}
    0 & 1 \\\
    -k/m & -c/m
    \end{bmatrix}S(t)
\end{aligned}
\end{equation}
$$

So the second order ODE describes motion of mass \\(m\\) is transformed into a first order ODE of the state \\(S(t)\\).


## Numerical methods of solving first order ODE

Given the formulation \\(\frac{dS}{dt} = \mathcal{F}(t, S(t))\\), and a regular grid on temporal interval \\([0, T]: \\{t_0, t_1,\cdots t_N\\}\\), where \\(t_i = i\frac{T}{N}=:ih\\). The Taylor expansion of \\(S\\) about \\(t_i\\) is given by: 

$$
\begin{equation}
\begin{aligned}
    S(t_{i+1}) = S(t_i) + \sum_1^k{
        \frac{h^k}{k!}
        \frac{d^{(k)}S}{dt}(t_i)
    } + \mathcal{O}(h^{k+1})
\end{aligned}
\end{equation}
$$

### Euler method
The Euler method approximate the next state by simply truncating the Taylor expansion after the first derivative.

$$
\begin{equation}
    \begin{aligned}
    \hat{S}(t_{i+1}) & = S(t_i) + h \frac{dS}{dt}(t_i) + \mathcal{O}(h^2)\\\
    & = S(t_i)  + h\mathcal{F}(t, S(t)) + \underbrace{\red{\mathcal{O}(h^2)}}\_{\text{Truncation Error}}
    \end{aligned}
\end{equation}
$$


Given \\(\mathcal{F}, S_0\\) we can sequentially compute \\(S\\) at any time \\(t\\).

{{< collapse summary="**(code) JAX Implementation of Euler method `euler()`**" >}} 
```python
def euler(z0: jnp.ndarray, t0: float, t1: float, f: callable,
          return_seq: bool = False):
    n_steps = int(jnp.ceil(jnp.abs(t1 - t0)/H_MAX))

    # Compute step size
    h = (t1 - t0)/n_steps

    t = t0
    z = z0

    # sequence of z
    seq = [(z, t)]

    for i in range(n_steps):
        z = z + h * f(z, t)
        t = t + h

        seq.append((z, t))

    if return_seq:
        return z, seq

    return z

```
{{< /collapse >}}

Let's try this on the dampen harmonic occilliator example with parameters:

```yml
c: .1       # Dampener coef
k: 1.       # Spring coef
m: 1.       # mass
A: 1.       # Initial position
V: -1.      # Inital velocity
t0: .0      # Start time
t1: 100.    # Terminal time
```

{{< collapse summary="**(code) `dampen_harmonic_occiliator()`**" >}} 
```python
# Dampen Harmoic Occililator
def dampen_harmonic_occiliator(
        c   : PositiveFloat,
        k   : PositiveFloat,
        m   : PositiveFloat,
        A   : float,
        V   : float,
        t0  : float,
        t1  : float):
    """
    Problem: 
        ODE: x'' + c/m x' + k/m x = 0
        IV: x(t0) = A, x'(t0) = V

    Params:
        c: dampener coefficient
        k: spring coefficient
        m: mass
        A, V: initial position and velocity
        t0, t1: start and terminal timestamp
    """

    # constructing dynamic function
    F = lambda S, t: jnp.array([[0, 1],[-k/m, -c/m]]) @ S

    # initial condition
    S = jnp.array([A, V])

    # Terminal S and trajectory of S
    S_t, Tr = euler(S, t0, t1, F, True)

    return S_t, Tr

```
{{< /collapse >}}


```python
# Terminal state & Trajectory of S
S_T, Tr = dampen_harmonic_occiliator(
        config.c,
        config.k,
        config.m,
        config.A,
        config.V,
        config.t0,
        config.t1,)

# Position
X = [i[0][0] for i in Tr]
T = [i[1] for i in Tr]

plt.plot(T, X)
plt.show()

```

#### Results

![](/images/ode_euler_rs.png)

### Runge-Kutta


## References
- (Book) [Partial Differential Equations for Scientists and Engineers - Standley J. Farlow](https://www.amazon.com/Differential-Equations-Scientists-Engineers-Mathematics/dp/048667620X)
- (Book) [Python Programming and Numerical Methods - A Guide]() - Chapter 22
