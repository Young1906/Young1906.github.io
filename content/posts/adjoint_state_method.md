---
title: Understanding Adjoint sensitivity method 
date : 2024-10-01
tags : [learn]
draft: true 
categories: [
    "Machine Learning",
    "PDE",
    "Numerical methods",
    ]
comments: true 
---


# Problem


$$
\begin{equation}
\begin{aligned}
\text{ODE:}\quad &\begin{cases} \frac{du}{dt} &= f(u, t; \theta) &\text{(Dynamic function)}\\\
   u(t=0) &= u_0  & \text{(Initial value)}
   \end{cases}
\end{aligned}
\end{equation}
$$




Minimize loss functional:

$$
\begin{equation}
\begin{aligned}
\mathcal{J}[\theta] = \int_{0}^T{g(u; \theta)} dt
\end{aligned}
\end{equation}
$$

## Formulating as an optimization with ODE constraint:

$$
\begin{equation}
\begin{aligned}
\min_\theta \mathcal{J}(\theta) = \min_\theta \int_0^Tg(u; \theta) dt \\\
\text{such that: } \\
\frac{du}{dt} - f(u, t;\theta) = 0 
\end{aligned}
\end{equation}
$$

<!--
## Application

### Classification problem

$$
\begin{equation}
\begin{aligned}
\text{softmax}(\mathcal{l}_\theta(y(t=T)))
\end{aligned}
\end{equation}
$$

-->

# Derivation of the adjoint-state



The Lagrangian of the optimization problem

$$
\begin{equation}
\begin{aligned}
\mathcal{L}(u, \lambda;\theta) & = 
    \mathcal{J}(\theta) +
    \underbrace{
        \int_0^T{\lambda^\top(t)(f - \frac{du}{dt}) dt}
    }_{=0 \text{ due to the ODE}} \\\
& = \int_0^T{
    \big[g(u;\theta) + \lambda^\top(t)(f - \frac{du}{dt})\big]dt
}
\end{aligned}
\end{equation}
$$


We need to compute the the total derivative of loss functional with respect to (w.r.t) \\(\theta\\), differentiating *L.H.S* and *R.H.S* w.r.t to \\(\theta\\):

$$
\begin{equation}
\begin{aligned}
\frac{d\mathcal{L}}{d\theta} & = \frac{d}{d\theta}\bigg(\int_0^T{
    \big(g(u;\theta) + \lambda^\top(t)(f - \frac{du}{dt})\big)dt
}\bigg)\\\
& = \int_0^T{
\frac{d}{d\theta}
    \big(g(u;\theta) + \lambda^\top(t)(f - \frac{du}{dt})\big)dt
} \\\
& = \int_0^T {\big\(
\frac{\partial g}{\partial \theta} + \frac{\partial g}{\partial u}\red{\frac{du}{d\theta}}
    + \lambda^\top(t)(
        \frac{\partial f}{\partial \theta} 
        + \frac{\partial f}{\partial u}\red{\frac{du}{d\theta}}
        - \frac{d}{d\theta}\frac{du}{dt}
    )
\big)dt} \\\
& = \int_0^T{\big(
    \frac{\partial g}{\partial\theta} + \lambda^\top(t) \frac{\partial f}{\partial \theta}
    + (\frac{\partial g}{\partial u} + \lambda^\top(t)\frac{\partial f}{\partial u})\red{\frac{du}{d\theta}} 
    - \lambda^\top(t) \frac{d}{dt}\red{\frac{du}{d\theta}}
\big)dt}\\\
& = \int_0^T{\big(
    \frac{\partial g}{\partial\theta} + \lambda^\top(t) \frac{\partial f}{\partial \theta}
    + (\frac{\partial g}{\partial u} + \lambda^\top(t)\frac{\partial f}{\partial u})\red{\frac{du}{d\theta}}
    \underbrace{
    - \lambda^\top(t) \frac{d}{dt}\red{\frac{du}{d\theta}
    }}\_{A}
\big)dt}
\end{aligned}
\end{equation}
$$

> **Note**: Initially I understood that bringing the differentiation into the integral is result of Leibniz Integral Rule. However, in the Leibniz Rule \\( dJ/dx = d/dx \int_a^b{f(x, t)} dt = \int_a^b{\partial/\partial x f(x, t) dt}\\), the total deriviative is changed to partial derivative because \\(x\\) is not a function of \\(t\\) in constrast to our case where \\(u\\) is a function of \\(\theta\\). Refer to [this paper](https://www.sciencedirect.com/science/article/pii/S1053811914003097) for derivation.

Consider the integration of term \\(A\\):

$$
\begin{equation}
\begin{aligned}
\int_0^T {
        -\lambda^\top(t) \frac{d}{dt}\red{\frac{du}{d\theta}} dt 
} &= \bigg[-\lambda^\top(t)\red{\frac{du}{d\theta}}\bigg]_0^T + \int_0^T{\frac{d\lambda}{dt}^\top \red{\frac{du}{d\theta}}dt} \\\
&= \lambda^\top(0) \red{\frac{du}{d\theta}}\big\vert\_{t=0} - \lambda^\top(T)\red{\frac{du}{d\theta}}\big\vert\_{t=T} + \int_0^T{\frac{d\lambda}{dt}^\top \red{\frac{du}{d\theta}}dt} \\\
\end{aligned}
\end{equation}
$$

Replacing result from eq. (6) into eq. (5):

$$
\begin{equation}
\begin{aligned}
    \frac{d\mathcal{L}}{d\theta} 
    &= \int_0^T {\big(
        \frac{\partial g}{\partial\theta} + \lambda^\top(t)\frac{\partial f}{\partial \theta}
        + \underbrace{
            (\frac{\partial g}{\partial u} + \lambda^\top(t) \frac{\partial f}{\partial u} - \frac{d\lambda}{dt}^\top)\red{\frac{du}{d\theta}}}\_{B}
    \big)dt} \\\
        & + \lambda^\top(0)\red{\frac{du}{d\theta}}\big\vert_{t=0}  -
        \underbrace{\lambda^\top(T)\red{\frac{du}{d\theta}}\big\vert_{t=T}}_{C}
\end{aligned}
\end{equation}
$$


Because the Jacobian \\(\frac{du}{d\theta}\\) is computationally expensive, we can choose \\(\lambda(t)\\) such that \\(A\\) and \\(B\\) vanish from equation (7) and compute \\(\lambda^\top(0)\\) by solving terminal value ODE: 

> See section **Forward sensitivity method** bellow for directly computing \\(\red{\frac{du}{d\theta}}\\)


$$
\begin{equation}
\begin{aligned}
& \begin{cases}
\frac{\partial g}{\partial u} + \lambda^\top(t) \frac{\partial f}{\partial u} - \frac{d\lambda}{dt}^\top = 0 \\\
\lambda^\top(T) = 0
\end{cases} \\\
\implies & \begin{cases}
\frac{d\lambda}{dt}^\top = \frac{\partial g}{\partial u} + \lambda^\top(t) \frac{\partial f}{\partial u} \\\
\lambda^\top(T) = 0
\end{cases} & \text{\small(Rearrange)}\\\
\implies & \begin{cases}
\frac{d\lambda}{dt} = \frac{\partial g}{\partial u}^\top + \frac{\partial f}{\partial u}^\top\lambda \\\
\lambda(T) = 0
\end{cases} & \text{\small(Transposition both L.H.S and R.H.S)}
\end{aligned}
\end{equation}
$$


Then the gradient of loss functional w.r.t network parameters becames:

$$
\begin{equation}
\begin{aligned}
\frac{d\mathcal{L}}{d\theta} &= \int_0^T {\big(
    \frac{\partial g}{\partial_\theta} + \lambda^\top(t)\frac{\partial f}{\partial \theta}
\big)dt} + \lambda^\top(0) \frac{du}{d\theta}\big\vert_{t=0}
\end{aligned}
\end{equation}
$$

\\(\lambda(t)\\) is called the adjoint-state.


### Computing \\(\frac{du}{d\theta}\big\vert_{t=0}\\)
### Computing \\(\frac{du}{d\theta}\big\vert_{t=0}\\)




## Forward sensitivity method
TBD

# Some examples

## Wolf and Bunny population dynamic 

Let \\(b(t)\\) and \\(w(t)\\) be population function of time for bunnies and wolves respectively. The change in population can be described by a coupled ODE

$$
\begin{equation}
\begin{aligned}
\text{ODEs: } & \begin{cases}
\frac{dr}{dt} = 4 r(t) - 2w(t) \\\
\frac{dw}{dt} = r(t) + w(t)
\end{cases} \\\
\text{Initial values: } & r(0) = 100,\quad  t(0) = 2
\end{aligned}
\end{equation}
$$


Let \\(S(t) = \begin{bmatrix}b(t)\\\s(t)\end{bmatrix}\\) be a vector-valued function of time describe wolves and bunnies population. Eq. (10) can be rewritten as:

$$
\begin{equation}
\begin{aligned}
\text{ODE:} & \frac{dS}{dt} = \begin{bmatrix}
4 & -2 \\\
1 & 1
\end{bmatrix} S(t)\\\
\text{Initial value: } & S(0) = \begin{bmatrix}
100\\\
2
\end{bmatrix}
\end{aligned}
\end{equation}
$$

# Appendix

**Table of notations**

|
:---: | :--- 
\\(\vec{u}(t) \in \mathbb{R}^N \\) | Solution function to initial value ODE (1)
\\(\theta \in \mathbb{R}^P \\) | Collection of parameters 
\\(g(t): \mathbb{R}^N \rightarrow \mathbb{R}\\) | Some loss function (i.e, MSE)
\\(f(u, t; \theta)\\\ f : \mathbb{R}^{N \times P} \rightarrow \mathbb{R}^N \\) | Dynamic function parameterized by \\(\theta\\), describes the gradient field of state \\(u\\) given its current location 
\\(\mathcal{J}: \mathcal{F}\rightarrow \mathbb{R}\\) | Loss functional, mapping from loss function onto real number line.


# References


- [Efficient gradient computation for dynamical models](https://www.sciencedirect.com/science/article/pii/S1053811914003097)
