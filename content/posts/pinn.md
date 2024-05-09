---
title: Learning physics-informed Neural Networks (PINN)
date : 2024-05-09
tags : [learn, ml]
draft: True 
categories: [
    "Machine Learning",
    "PDE",
    "Learning"
    ]
---


My goal for this post is to familiarize myself with the paradigm of PINN, consist of ... Having no formal training on solving PDE, the first section will outline basic concepts in solving partial differential equations (PDEs), an example of a PDE, and its closed-form solution.


Outline:
- PDE
- Closed-formed solution
- Generating data
- PINN

## PDE and the Legendre Differential Equation

This section is written with instruction from [this introduction](https://web.math.ucsb.edu/~moore/pde.pdf) by JD Moore.


<!--
$$
    (1 - x^2) \frac{d^2y}{dx^2} - 2x\frac{dy}{dx} + p(p+1) y = 0
$$

Where \\(x \in (-1, 1) \\), and \\(p\\) is a parameter.
-->


**Definition** (Power series) 

$$
\sum_{n=0}^\infty{a_n(x - x_0)^n}
$$

**Theorem** For any power series centered at \\(x_0\\), there exists a non-negative real number \\(R\\) or \\(\infty\\), such that
1. The power series converges when \\(|x - x_0| < R\\)
1. and diverges when \\(|x - x_0| > R\\)

We refer \\(R\\) as *radius of convergence*.

> **Note** 

**Definition** (Ratio test) The radius of convergence of a power series

$$
\sum_{n=0}^\infty{a_n(x - x_0)^n}
$$

is given by the formula

$$
R = \lim_{n\rightarrow \infty} \frac{|a_n|}{|a_{n+1}|}
$$


**Definition** (Comparison test) Suppose that the power series

$$
\begin{aligned}
\sum_{n=0}^\infty{a_n(x - x_0)^n},\quad \sum_{n=0}^\infty{b_n(x - x_0)^n}
\end{aligned}
$$


## References
- [Introduction to Partial Differential Equation - John Douglas Moore](https://web.math.ucsb.edu/~moore/pde.pdf)



