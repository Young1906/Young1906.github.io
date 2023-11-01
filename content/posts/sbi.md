---
title: Simulation Based Inference
date : 2023-10-31
tags : [sbi]
draft: true
---


## Simulation Based Inference

Imagine we have some black-box machine, such machine has some knobs and levels so we can change its inner configurations. The machine churn out some data for each configuration. The **simulation based inference (SBI)** solve the inverse problem; that is given some data, estimate the configuration, or what are the most likely value of the configuration. 

In this context, the machine is called a simulator, the configuration is called the parameters and the data is called observation. A simulator can be abstraction of many 



### Problem setup
There are four components in the problem of **Simulation Based Inference** (SBI), namely the simulator, the parameters, the latent variables, and the observations. The simulator implicitly define the statistical model \\(p(x | \theta) = \int_z\\)


## Likelihood Ratio Estimation

Likelihood ratio is defined as the ratio between the likelihood of the observation between two different hypothesis:

$$
r(x | \theta_0, \theta_1) = \frac{p(x | \theta_0)}{p(x|\theta_1)}
$$

This quantity then can be used in various methods to draw sample from a distribution. In the paper, the author mention three sampling methods, namely **Markov Chain Monte Carlo**, **Metropolis-Hasting**, and HMC. In the following section, I am briefly summarizing those methods.

### Background

#### Markov Chain Monte Carlo (MCMC)

In MCMC sampling method, we assumed to have access to likelihood function \\(p(x)\\). To sample \\(x\\), we use a symmetric proposal distribution  \\( q(x_t | x_{t-1}) = q(x_{t-1} | x_t)\\)


#### Metropolis-Hasting (MH) 


### Method


### Toy example
![img](/images/amcmc.png)

## Reference

- [The frontier of simulation-based inference](https://www.pnas.org/doi/full/10.1073/pnas.1912789117)
