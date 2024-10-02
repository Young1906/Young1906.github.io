---
title: Neural Odinary Differential Equation - NeuralODE
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


We need to compute the the total derivative of loss functional with respect to (w.r.t) \\(\theta\\):

$$
\begin{equation}
\begin{aligned}
\frac{d\mathcal{L}}{d\theta} & = \frac{d}{d\theta}\bigg(\int_0^T{
    \big(g(u;\theta) + \lambda^\top(t)(f - \frac{du}{dt})\big)dt
}\bigg)\\\
& = \int_0^T{
\frac{\partial}{\partial\theta}
    \big(g(u;\theta) + \lambda^\top(t)(f - \frac{du}{dt})\big)dt
} \quad \text{(Leibniz Integral Rule)}\\\
& = \int_0^T {\big\(
\frac{\partial g}{\partial \theta} + \frac{\partial g}{\partial u}\frac{du}{d\theta}
    + \lambda^\top(t)(
        \frac{\partial f}{\partial \theta} 
        + \frac{\partial f}{\partial u}\frac{du}{d\theta} 
        - \frac{d}{d\theta}\frac{du}{dt}
    )
\big)dt} \\\
& = \int_0^T{\big(
    \frac{\partial g}{\partial\theta} + \lambda^\top(t) \frac{\partial f}{\partial \theta}
    + (\frac{\partial g}{\partial u} + \lambda^\top(t)\frac{\partial f}{\partial u})\frac{du}{d\theta} 
    - \lambda^\top(t) \frac{d}{dt}\frac{du}{d\theta}
\big)dt}\\\
& = \int_0^T{\big(
    \frac{\partial g}{\partial\theta} + \lambda^\top(t) \frac{\partial f}{\partial \theta}
    + (\frac{\partial g}{\partial u} + \lambda^\top(t)\frac{\partial f}{\partial u})\frac{du}{d\theta} 
    \red{
    - \lambda^\top(t) \frac{d}{dt}\frac{du}{d\theta}
    }
\big)dt}
\end{aligned}
\end{equation}
$$

Consider the integration of the last term:

$$
\begin{equation}
\begin{aligned}
\int_0^T {
        -\lambda^\top(t) \frac{d}{dt}\frac{du}{d\theta} dt 
} &= \bigg[-\lambda^\top(t)\frac{du}{d\theta}\bigg]_0^T + \int_0^T{\frac{d\lambda}{dt}^\top \frac{du}{d\theta}dt} \\\
&= \lambda^\top(0) \frac{du}{d\theta}\big\vert\_{t=0} - \lambda^\top(T)\frac{du)}{d\theta}\big\vert\_{t=T} + \int_0^T{\frac{d\lambda}{dt}^\top \frac{du}{d\theta}dt} \\\
\end{aligned}
\end{equation}
$$

Replacing result from equation (6) into equation (5):

$$
\begin{equation}
\begin{aligned}
    \frac{d\mathcal{L}}{d\theta} 
    &= \int_0^T {\big(
        \frac{\partial g}{\partial\theta} + \lambda^\top(t)\frac{\partial f}{\partial \theta}
        + \underbrace{
            (\frac{\partial g}{\partial u} + \lambda^\top(t) \frac{\partial f}{\partial u} - \frac{d\lambda}{dt}^\top)\frac{du}{d\theta}
        }\_{A}
    \big)dt} \\\
        & + \lambda^\top(0)\frac{du}{d\theta}\big\vert_{t=0}  -
        \underbrace{\lambda^\top(T)\frac{du}{d\theta}\big\vert_{t=T}}_B
\end{aligned}
\end{equation}
$$


Because the Jacobian \\(\frac{du}{d\theta}\\) is computationally expensive, we can choose \\(\lambda(t)\\) such that \\(A\\) and \\(B\\) vanish from equation (7) and compute \\(\lambda^\top(0)\\) by solving terminal value ODE: 

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

# Appendix

**Table of notations**

|
--- | --- 
\\(\vec{u}(t) \in \mathbb{R}^N \\) | Solution function to initial value ODE (1)
\\(g(t): \mathbb{R}^N \rightarrow \mathbb{R} \\) | Some loss function (i.e, MSE)
\\(f(u, t; \theta): \mathbb{R}^{N \times P} \rightarrow \mathbb{R}^N \\) | Dynamic function parameterized by \\(\theta\\), describes the gradient field of state \\(u\\) given its current location 
\\(\mathcal{J}: \mathcal{F}\rightarrow \mathbb{R}\\) | Loss functional, mapping from loss function onto real number line.

