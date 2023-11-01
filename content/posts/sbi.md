---
title: Simulation Based Inference
date : 2023-10-31
tags : [sbi]
draft: true
categories: [
    "Machine Learning",
    ]
---


## Simulation Based Inference

Imagine we have some black-box machine, such machine has some knobs and levels so we can change its inner configurations. The machine churn out some data for each configuration. The **simulation based inference (SBI)** solve the inverse problem; that is given some data, estimate the configuration, or what are the most likely value of the configuration. 

In this context, the machine is called a simulator, the configuration is called the parameters and the data is called observation. A simulator can be abstraction of many 



### Problem setup
There are four components in the problem of **SBI**, namely the simulator, the parameters, the latent variables, and the observations. The simulator implicitly define the statistical model \\(p(x | \theta) = \int_z{p(x, z | \theta) dz}\\), this quantity not just intractable but also unknown.


## Likelihood-free MCMC with Amortized Ratio Estimator

Likelihood ratio is defined as the ratio between the likelihood of the observation between two different hypothesis:

$$
r(x | \theta_0, \theta_1) = \frac{p(\mathbf{x} | \theta_0)}{p(\mathbf{x}|\theta_t)}
$$

This quantity then can be used in various methods to draw sample from a distribution. In the paper, the author mention three sampling methods, namely **Markov Chain Monte Carlo**, **Metropolis-Hasting**, and HMC. In the following section, I am briefly summarizing those methods.

### Background

#### Markov Chain Monte Carlo (MCMC)

We want to sample from \\(p(\theta | \mathbf{x})\\) using MCMC, we need this quantity

$$
\begin{equation}
\begin{aligned}
    \frac{p(\theta | \mathbf{x} )}{p(\theta_t| \mathbf{x})} = 
    \frac{
        p(\theta)p(\mathbf{x} | \theta)/p(\mathbf{x})
    }{
        p(\theta_t)p(\mathbf{x} | \theta_t)/p(\mathbf{x})
    } = 
    \frac{p(\theta)}{p(\theta_t)}\times
    \frac{p(\mathbf{x} | \theta)}{p(\mathbf{x} | \theta_t)}
\end{aligned}
\end{equation}
$$


#### Metropolis-Hasting (MH) 


### Likelihood Ratio Estimator 


### Toy example

- Setup


- Result
![img](/images/amcmc.png)



## Reference

- [The frontier of simulation-based inference](https://www.pnas.org/doi/full/10.1073/pnas.1912789117)
