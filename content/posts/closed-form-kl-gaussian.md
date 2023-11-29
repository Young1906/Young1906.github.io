---
title: Deriving closed-form Kullback-Leibler divergence for Gaussian Distribution 
draft: false 
date: 2023-11-12
tags: [learning, stat]
---

The closed form of KL divergence used in Variational Auto Encoder.

## Univariate case

Let 

- \\(p(x) = \mathcal{N}(\mu_1, \sigma_1) = (2\pi\sigma_1^2)^{-\frac{1}{2}}\exp[-\frac{1}{2\sigma_1^2}(x-\mu_1)^2]\\)
- \\(q(x) = \mathcal{N}(\mu_1, \sigma_2) = (2\pi\sigma_2^2)^{-\frac{1}{2}}\exp[-\frac{1}{2\sigma_2^2}(x-\mu_2)^2]\\)

KL divergence between \\(p\\) and \\(q\\) is defined as:

$$
\begin{aligned}
\text{KL}(p\parallel q) &= -\int_{x}{p(x)\log{\frac{q(x)}{p(x)}}dx} \\\
&= -\int_x p(x) [\log{q(x)} - \log{p(x)}]dx \\\
&= 
	\underbrace{
		\int_x{p(x)\log p(x) dx}}_A 
	- \underbrace{
	\int_x{p(x)\log q(x) dx}}_B
\end{aligned}
$$

First quantity \\(A\\):


$$
\begin{aligned}
	A &= \int_x{p(x)\log p(x) dx} \\\
	&= \int_x{p(x)\big[
	 -\frac{1}{2}\log{2\pi\sigma_1^2 
	 - \frac{1}{2\sigma_1^2}(x - \mu_1)^2}
    \big]dx}\\\
	&= -\frac{1}{2}\log{2\pi\sigma_1^2}\int_x{p(x)dx} 
		- \frac{1}{2\sigma_1^2} 
		\underbrace{\int_x{p(x)(x-\mu_1)^2dx}}_{\text{var(x)}}\\\
	&= -\frac{1}{2}\log{2\pi} - \log\sigma_1-\frac{1}{2}
\end{aligned}
$$
	

The second quantity \\(B\\):

$$
\begin{aligned}
    B =& \int_x{p(x)\big[
    	-\frac{1}{2}\log2\pi\sigma_2^2
    	- \frac{1}{2\sigma_2^2}(x-\mu_2)^2
    	\big]dx}\\\
	=& -\frac{1}{2}\log2\pi\sigma_2^2 
	- \frac{1}{2\sigma_2^2}\int_x{
 p(x)\big[
	 (x - \mu_1)^2 + 2(x-\mu_1)(\mu_1 - \mu_2) + (\mu_1 -\mu_2)^2
 \big]dx} \\\
 =& -\frac{1}{2}\log2\pi\sigma_2^2 \\\
 & - \frac{1}{2\sigma_2^2}\underbrace{\int_x{p(x)(x-\mu_1)^2}}_{\text{var}(x)}\\\
 & - \frac{2(\mu_1 -\mu_2)}{2\sigma_2^2} \underbrace{\int_x{p(x)(x-\mu_1)dx}}_0 \\\
 & - \frac{(\mu_1-\mu_2)^2}{2\sigma_2^2} \\\
 =& -\frac{1}{2}\log2\pi -\log\sigma_2 - \frac{\sigma_1^2}{2\sigma_2^2} - \frac{(\mu_1-\mu_2)^2}{2\sigma_2^2}
\end{aligned}
$$


Finally, we obtained the KL divergence for univariate case

$$
\begin{aligned}
    \text{KL}(p\parallel q) &= A - B \\\
&= (-\frac{1}{2}\log2\pi - \log\sigma_1 - \frac{1}{2}) - ( -\frac{1}{2}\log2\pi -\log\sigma_2 - \frac{\sigma_1^2}{2\sigma_2^2} - \frac{(\mu_1-\mu_2)^2}{2\sigma_2^2}) \\\
    &= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} + \frac{(\mu_1 -\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
\end{aligned}
$$

## Multivariate case

`tbd`
## Reference
- https://gregorygundersen.com/blog/
