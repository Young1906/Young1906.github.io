---
title: Learning physics-informed Neural Networks (PINN)
date : 2024-05-09
tags : [learn, ml, pde, pinn]
draft: True 
categories: [
    "Machine Learning",
    "PDE",
    "Learning"
    ]
---


My goal is to familiarize myself with physics-informed neural network(PINN), consist of ... Having no formal training on solving PDE, the first section will outline basic concepts in solving partial differential equations (PDEs), an example of a PDE, and its closed-form solution.


Outline:
- PDE
- Closed-formed solution
- Generating data
- PINN

## PDE and the Legendre Differential Equation

This section is written with instruction from [this introduction](https://web.math.ucsb.edu/~moore/pde.pdf) by JD Moore.


<!--
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

**Definition** (Singular point)

**Definition** (Regular singular point)

**Definition** (Ordinary singular point)

### Legendre differential equation

The Legendre differential equation [(2)](https://mathworld.wolfram.com/LegendreDifferentialEquation.html) is the second-order ordinary differential equation , which has the form:  

$$
    \begin{equation}
    (1 - x^2) \frac{d^2y}{dx^2} - 2x\frac{dy}{dx} + p(p+1) y = 0
    \end{equation}
$$

Where \\(p\\) is a parameter.

#### Solving Legendre differential equation

Assume \\(y\\) has the form of a power series centered at \\(0\\)

$$
\begin{aligned}
y &= \sum_{n=0}^\infty{a_n x^n}\\\
\frac{dy}{dx} &= \sum_{n=1}^\infty{n a_n x^{n-1}}\\\
\frac{d^2y}{dx^2} &= \sum_{n=2}^\infty{n(n-1)a_n x^{n-2}}\\\
\end{aligned}
$$

Subsitute into (eq. 1) we have:

$$
\begin{equation}
\begin{aligned}
& (1 - x^2)\sum_{n=2}^\infty{n(n-1)a_n x^{n-2}} - 2x \sum_{n=1}^\infty{n a_n x^{n-1}} + p(p+1)\sum_{n=0}^\infty{a_n x^n} &= 0 \\\
\implies & 
    \underbrace{
        \sum_{n=2}^\infty{n(n-1)a_n x^{n-2}}
    }\_{A}
    - \underbrace{
    \sum_{n=2}^\infty{n(n-1)a_n x^n}
    }\_{B}
    - 2
    \underbrace{
        \sum_{n=1}^\infty{n a_n x^n}
    }\_{C}
    + p(p+1)\sum_{n=0}^\infty{a_n x^n} & = 0
\end{aligned}
\end{equation}
$$


- Term \\(A\\):

$$
    \begin{aligned}
        A &= \sum_{n=2}^\infty{n(n-1)a_n x^{n-2}} \\\
        &= \sum_{m=0}^\infty (m+1)(m+2)a_{m+2} x^m & \text{(replacing } n = m + 2 \text{)}\\\
        &= \sum_{n=0}^\infty (n+1)(n+2)a_{n+2} x^n & \text{(because m is a dummy index)}
    \end{aligned}
$$

- Term \\(B\\)
$$
    \begin{aligned}
        B & = \sum_{n=2}^\infty{n(n-1)a_n x^n} \\\
        &= 0 (0 - 1) a_0 x^0 + 1 ( 1- 1) a_1 x^1 + \sum_{n=2}^\infty{n(n-1)a_n x^n}\\\
        &= \sum_{n=0}^\infty{n(n-1)x^n}
    \end{aligned}
$$

- Same trick with term \\(C\\)
$$
C = \sum_{n=0}^\infty{n a_n x^n}
$$

Substitute \\(A, B, C\\) into eq. 2, we have

$$
\sum_{n=0}^\infty{
    \big[
    \underbrace{
        (n+1)(n+2)a_{n+2} - n(n-1)a_n - 2na_n + p(p+1)a_n
    }_{=0}
    \big] x^n
} = 0
$$

The equality above is satisifed iff each coefficient of the polinomial is \\(0\\). So we can obtains the recursion form of coefficients

$$
a_{n+2} = \frac{n(n+1) - p(p+1)}{(n+1)(n+2)}a_n = \frac{(n-p)(p+n+1)}{(n+1)(n+2)}a_n
$$

Where \\(a_0, a_1\\) are given by initial conditions \\(y(0), \frac{dy}{dx}(0)\\).

- For even \\(n\\):
    - \\(n = 0, a_2 = -\frac{p(p+1)}{1\times 2}a_0 \\)
    - \\(n = 2, a_4 = -\frac{(p-2)(p+3)}{3 \times 4}a_2 = \frac{p(p+1)(p-2)(p+3)}{4!}a_0 \\)
    - \\(n = 4, a_6 = -\frac{(p-4)(p+5)}{5 \times 6}a_4  = -\frac{p(p+1)(p-2)(p+3)(p-4)(p+5)}{6!} a_0 \\)
    - \\(\cdots\\)

    So that

    $$
        a_{2k} = (-1)^{k+1} \frac{p(p+1)(p-2)\cdots (p-2k)(p+2k+1)}{(2k+2)!}
    $$

- Similary for odd \\(n\\)
    - \\(n = 1, a_3 = -\frac{(p - 1)(p+2)}{2\times 3}a_1 \\)
    - \\(n = 3, a_5 = -\frac{(p - 3)(p+4)}{4\times 5}a_3 = \frac{(p-1)(p+2)(p-3)(p+4)}{5!}a_1\\)
    - \\(n = 5, a_7 = -\frac{(p - 5)(p+6)}{6\times 7}a_5 = -\frac{(p-1)(p+2)(p-3)(p+4)(p-5)(p+6)}{7!}a_1 \\)
    - \\(\cdots\\)

    So that

    $$
        a_{2k+1} = (-1)^{k+1} \frac{(p-1)(p+2)\cdots (p-2k+1)(p+2k+2)}{(2k+1)!}
    $$
    
> TODO: 
> - Perform ratio test to analyze radius of convergence.
> - Analysis of singular points?

## Using PINN to solve PDE

### Construction / Components

- Dataset {X, y}
- Parameterize \\(y = f_\theta(x)\\) where \\(f\\) is a neural network.
- Define loss function: \\(\text{MSE} = \text{MSE}_f + \text{MSE}_e\\)


### Comparing PINN to ordinary Multi-Layer Perceptron (MLP) 
### Implemetation

#### Training
#### Evaluation

## References
- (1) [Introduction to Partial Differential Equation - John Douglas Moore](https://web.math.ucsb.edu/~moore/pde.pdf)
- (2) [https://mathworld.wolfram.com](https://mathworld.wolfram.com/LegendreDifferentialEquation.html)


---

## TODO:

- [ ] Understand the ratio test
