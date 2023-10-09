---
title: "Understanding diffusion model"
date: 2023-06-23T13:29:01+07:00
draft: true 
tags: study
---


## Motivation


## Background

### Generative model
### Kullber-Leiback divergence (KL-divergence)

**Definition** (Kullback-Leibler Divergence)

\begin{equation}
    \texttt{KL}(P \parallel Q) = -\int_{x \in \mathcal{X}}{p(x)\log{\frac{q(x)}{p(x)}}dx}
\end{equation}

### The Evidence Lower Bound (ELBO)

\begin{equation}
    \begin{aligned}
        p(x) 
        &= \int_z{p(x, z)dz}\\
        &= \int_z{p(x\vert z)p(z)dz}
    \end{aligned}
\end{equation}



### Variational Auto Encoder
### Hierachical Variational Encoder

## Denoising Diffusion Probabilistics Model

## Implementation
