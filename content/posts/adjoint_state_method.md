---
title: Understanding Adjoint sensitivity method 
date : 2024-10-01
tags : [learn]
draft: false 
categories: [
    "Machine Learning",
    "PDE",
    "Numerical methods",
    ]
comments: true 
---


# Note

$$
\red{\text{==================editing====================}} 
$$

Suppose we have a dataset \\(\{(t_i, u\_{t_i})\}\_{i=0\cdots N-1}, \quad u_{t_i} \in \mathbb{R}^N\\) is the observed state of a dynamical system given by ODE

\begin{equation}
	\begin{aligned}
		\begin{cases}
			u(t=t_i) = u_{t_i} \\
			\frac{du}{dt} = f(u, t, \theta)
		\end{cases}
	\end{aligned}
\end{equation}


For simplicity, assume we only have 2 observed states \\((t_0, u_0), (t_1, u_1)\\). So that we can write \\(u_1\\) in term of \\(u_0\\) and the dynamic \\(f\\)

$$
\begin{equation}
	\begin{aligned}
		u(t_1) = u_0 + \int_{t_0}^{t_1}{f(u, t, \theta) dt}
	\end{aligned}
\end{equation}
$$


Minimize the loss function \\(g: \mathbb{R}^N \rightarrow \mathbb{R}; g(u(t_1))\\) such that \\(\frac{du}{dt} = f;\quad \forall t \in [t_0, t_1]\\)


We can write the constrainted optimization problem as unconstrainted one using Lagrangian multiplier $\lambda(t)$

$$
\begin{equation}
	\begin{aligned}
		L(u(t_1), \lambda, \theta) = g(u(t_1)) + \int_{t_0}^{t_1}{\lambda(t)(f - \frac{du}{dt} )dt}
	\end{aligned}
\end{equation}
$$

We need to compute the total derivative $\frac{dL}{d\theta}$ in order to minimze functional $L$


$$
\begin{equation}
	\begin{aligned}
		\frac{dL}{d\theta} & = \frac{d}{d\theta}\bigg(
			g(u(t_1)) + \int_{t_0}^{t_1}{\lambda(t)(f - \frac{du}{dt} )dt}
		\bigg) \\\
		& = \frac{d}{d\theta} g(u(t_1), \theta) + \int_{t_0}^{t_1}{\frac{d}{d\theta}\big(
			\lambda(t)(f - \frac{du}{dt})
		\big) dt} \\\
		& = \frac{\partial g}{\partial u(t_1)} \frac{d u(t_1)}{d\theta}
		+ \int_{t_0}^{t_1}{\lambda(t)\frac{d}{d\theta}\big(
			f - \frac{du}{dt}
		\big) dt} \\\
		& = \frac{\partial g}{\partial u(t_1)} \frac{d}{d\theta}\bigg(
			u_0 + \int_{t_0}^{t_1}{f(u, t, \theta) dt}
		\bigg) + \int_{t_0}^{t_1}{\lambda(t)\frac{d}{d\theta}\big(
			f - \frac{du}{dt}
		\big) dt} \\\
		& = \frac{\partial g}{\partial u(t_1)} \int_{t_0}^{t_1}{\big(
			\frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial u} \frac{du}{d\theta}
		\big)dt} + 
		\int_{t_0}^{t_1}{\lambda(t)\big(
			\frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial u} \frac{du}{d\theta}
			- \frac{d}{d\theta}\frac{du}{dt}
		\big) dt} \\\
		& = \int_{t_0}^{t_1}{
			\bigg(
			\frac{\partial g}{\partial u(t_1)} \frac{\partial f}{\partial \theta}
			+ \frac{\partial g}{\partial u(t_1)} \frac{\partial f}{\partial u}\frac{du}{d\theta}
			+ \lambda(t) \frac{\partial f}{\partial \theta}
			+ \lambda(t)\frac{\partial f}{\partial u}\frac{du}{d\theta}
			\underbrace{
				- \lambda(t) \frac{d}{dt}\frac{du}{d\theta}
			}_{A}
			\bigg) dt
		}
	\end{aligned}
\end{equation}
$$

Consider the integral of term $A$


$$
\begin{equation}
	\begin{aligned}
		\int_{t_0}^{t_1}{- \lambda(t) \frac{d}{dt}\frac{du}{d\theta} dt} 
        = [\lambda(t)\frac{du}{d\theta}]_{t_1}^{t_0} + \int\_{t\_0}^{t\_1}{\frac{d\lambda}{dt}\frac{du}{d\theta}dt}
 \end{aligned}
\end{equation}
$$


Replace eq.(\ref{eq:term_a}) into eq.(\ref{eq:sensitivity}) we have:

$$
\begin{equation}
	\begin{aligned}
		\frac{dL}{d\theta} &= 
		\int_{t_0}^{t_1} {\bigg(
			\frac{\partial g}{\partial u(t_1)} + \lambda(t)
		\bigg)\frac{\partial f}{\partial \theta}dt} \\\
		& + \int_{t_0}^{t_1} {\bigg(
			\frac{\partial g}{\partial u(t_1)}\frac{\partial f}{\partial u}
			+ \lambda(t) \frac{\partial f}{\partial u} + \frac{d\lambda}{dt}
		\bigg)\red{\frac{du}{d\theta}}dt} \\\
		& + \lambda(t_0)\frac{du(t_0)}{d\theta} - \red{\lambda(t_1)\frac{du(t_1)}{d\theta}}
	\end{aligned}
\end{equation}
$$


Because $\frac{du}{d\theta}$ is computationally expensive, we can choose $\lambda(t)$ so that the second and forth term vanish so that we have another ODE

$$
\begin{equation}
	\begin{aligned}
		\begin{cases}
			\frac{d\lambda}{dt} = -\big(
			\frac{\partial g}{\partial u(t_1)} + \lambda(t)
			\big)\frac{\partial f}{\partial u} \\\
			\lambda(t_1) = \vec{0}
		\end{cases}
	\end{aligned}
\end{equation}
$$ 

Let $a(t) = \lambda(t) + \frac{\partial g}{\partial u(t_1)}$, eq.(\ref{eq:adjoint_ode}) becomes:

$$
\begin{equation}
	\begin{aligned}
		\begin{cases}
			\frac{da}{dt} = -a(t)\frac{\partial f}{\partial u} \\\
			a(t_1) = \frac{\partial g}{\partial u(t_1)}
		\end{cases}
	\end{aligned}
\end{equation}
$$


And eq.(\ref{eq:sensitivity_2}) becomes:

\begin{equation}
	\begin{aligned}
		\frac{dL}{d\theta} = 
		\big(a(t_0) - \frac{\partial g}{\partial u(t_1)}\big)\frac{du(t_0)}{d\theta}
		+ \int_{t_0}^{t_1}{a(t)\frac{\partial f}{\partial\theta} dt} 
	\end{aligned}
\end{equation}



# References


- [Efficient gradient computation for dynamical models](https://www.sciencedirect.com/science/article/pii/S1053811914003097)
