---
title: Simulation Based Inference - Likelihood Ratio Estimation
date : 2023-10-31
math: true
tags : [sbi]
---

# Background

Something

## Inference

### Simulation Based Inference

### Problem setup


There are four components in the problem of **Simulation Based Inference** (SBI), namely the simulator, the parameters, the latent variables, and the observations. 

- In the context of SBI, the simulator is a computer program taking parameters $\theta$, sample a series of latent states $z_i \sim p(z_i | \theta, z < i)$, and finally produces some observations $x \sim p(x |\theta, z)$. More concretly, the simulator implicitly define the likelihood function $p(x | \theta) = \int_z{p(x, z | \theta)dz}$. These quantity is not just intractable but also unknown.
- The parameter $\theta$
- The latent variables $z$
- The observations $x$

The goal of SBI is to samples from posterior distribution $p(\theta | x)$


## Likelihood Ratio Estimation


- [Likelihood-free MCMC with Amortized Approximate Likelihood Ratios]()
- [On Contrastive Learning for Likelihood-free Inference]()


### Likelihood-free MCMC with Amortized Approximate Likelihood Ratio

#### Background

##### Markov Chain Monte Carlo

##### Metropolis-Hasting

##### 



