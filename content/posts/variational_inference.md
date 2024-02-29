---
title: Understanding Variational Inference 
date : 2024-02-29
tags : [variational-inference]
draft: false 
categories: [
    "Machine Learning",
    ]
---


This post is a note I take from while reading [Blei et al 2018](). 

Goal:

- Motivation of variational inference
- Understand the derivation of ELBO and its intiution
- Walk through the derivation, some of which was skip the in original paper
- Implementation of CAVI





## ELBO

Goal is to find \\(q(z)\\) to approximate \\(p(z|x)\\)

The KL-divergence

$$
\begin{equation}
\begin{aligned}
	KL[q(z)||p(z | x)] &= 
		\int_z{q(z)\log{\frac{p(z|x)}{q(z)}} dz}
\end{aligned}
\end{equation}
$$

However, this quantity is intractable to compute hence, we're unable to optimize this quantity directly.

$$
\begin{equation}
\begin{aligned}
	KL[q(z)||p(z | x)] &= 
		- \int_z{q(z)\log{\frac{p(z|x)}{q(z)}} dz} \\\
		&= -\int_z{
			q(z) \log {
				\frac{\log p(z, x)}{q(z) p(x)}
			}
		}\\\
		&= -\int_z{q(z)[\log{\frac{p(z,x)}{q(z)}} - \log p(x)]dz} \\\
		&= -\int_z{
			q(z) \log \frac{p(z, x)}{q(z)}dz
		} + \int_z{q(z)\log p(x) dz} \\\
		& =: -\texttt{ELBO}[q] + \log p(x) \\\
	\iff \texttt{ELBO}[q] &= -KL(q||p) + \log p(x)
\end{aligned}
\end{equation}
$$


Because \\(\log p(x)\\) is a constant, by maximizing \\(\text{ELBO}[q]\\), we minimize \\(KL(q||p)\\) by proxy. Rewrite ELBO:

$$
\begin{equation}
\begin{aligned}
\texttt{ELBO}(q) &= \int_z{q(z)\log \frac{p(z, x)}{q(z)}} \\\
    &= \mathbb{E}\_{z\sim q}[\log p(z, x)] - \mathbb{E}\_{z\sim q}[\log q(z)]
\end{aligned}
\end{equation}
$$


## Mean field Variational Family

Mean-field variational family made a strong assumption of independence between it's latent variable

$$
q(\mathbf{z}) = \prod_{j} {q_j(z_j)}
$$

Coordinate ascent variational inference is a common method to solve mean-field variational inference problem. Holding other latent variable fixed, the \\(j^{th}\\) latent variable is given by:

$$
q^*_{j}(z_j) = \text{exp}{\mathbb{E}\_{-j}[\log p(z\_j | z\_{-j}, \mathbf{x})]} \propto \exp{\mathbb{E}\_{-j} [\log p(z\_j, z\_{-j}, \mathbf{x})]}
$$
		

### Proof


$$
\begin{equation}
\begin{aligned}
	q^*\_j(z_j) &= \texttt{arg}\max\_{q_j(z_j)} \quad{\texttt{ELBO}(q)} \\\
                &= \texttt{arg}\max\_{q_j(z_j)} \quad \mathbb{E}\_q[\log p(z_j, z_{-j}, x)] - \mathbb{E}\_q[\log q(z_j, z_{-j})] \\\
                &= \texttt{arg}\max\_{q_j(z_j)} \quad \mathbb{E}\_j[\mathbb{E}\_{-j}[\log p(z_j, z_{-j}, x)]] - \mathbb{E}\_j[\mathbb{E}\_{-j}[\log q_j(z_j) + \log q_{-j}(z_{-j})]] \\\
                &= \texttt{arg}\max\_{q_j(z_j)} \quad \mathbb{E}\_j[\mathbb{E}\_{-j}[\log p(z_j, z_{-j}, x)]] - \mathbb{E}\_j[\log q_j(z_j)] + const \\\
                &= \texttt{arg}\max\_{q_j(z_j)} \quad \mathbb{E}\_j[\mathbb{E}\_{-j}[\log p(z_j, z_{-j}, x)]] - \mathbb{E}\_j[\log q_j(z_j)]
\end{aligned}
\end{equation}
$$
	


We need to find function \\(q_j(z_j)\\) that maximize \\(\text{ELBO}(q)\\)

Assuming \\(q_j(z_j)= \epsilon \eta(z_j) + q^*_j(z_j)\\)

$$
\begin{aligned}
K(\epsilon) 
    &= \mathbb{E}\_j[\mathbb{E}\_{-j}[\log p(z_j, z\_{-j}, x)]] - \mathbb{E}\_j[\log q_j(z_j)] \\\
    &= \int_{z_j} q_j(z_j) A d_{z_j} - \int_{z_j}q_j(z_j)\log q_z(z_j) d_{z_j} \\\
    &= \int_{z_j} [\epsilon \eta(z_j) + q^*\_j(z_j)] A d_{z_j} - \int_{z_j}[\epsilon \eta(z_j) + q^\*\_j(z_j)] \log [\epsilon \eta(z_j) + q^\*\_j(z_j)] d\_{z_j}
\end{aligned}
$$


Evaluate the partial derivative of \\(K\\) wrt \\(\epsilon\\) we have:


$$
\begin{aligned}
    & \frac{\partial}{\partial \epsilon}K \bigg\vert_{\epsilon=0} = 0 \\\
    \iff & \int_{z_j} {\eta(z_j) A d\_{z_j}} - 
    	\int_{z_j} {
    	{\eta(z_j) \log [\epsilon \eta(z_j) + q^\*\_j(z_j)]} 
    	+ [\epsilon \eta(z_j) + q^\*\_j(z_j)] \frac{\eta(z_j)}{\epsilon \eta(z_j) + q^\*\_j(z_j)}d_{z_j} 
\} = 0\\\
    \iff & \int_{z_j} {\eta(z_j) A d_{z_j}} - \int_{z_j}{[\eta(z_j)\log q^\*\_j(z_j) +\eta(z_j)]d_{z_j}} = 0; \quad \forall \eta(z_j) \\\
    \iff & \log q^\*\_j(z_j) = A-1 = \mathbb{E}\_{-j}[\log p(z_j, z_{-j}, x)] - 1 \\\
    \iff & q^\*\_j(z_j) \propto \exp\{\mathbb{E}\_{-j}[\log p(z_j, z_{-j}, x)]\}
\end{aligned}
$$

## Complete example of Bayesian Gaussian Mixture
TDB

<!--
Assuming following latent variable
- Cluster assignment variable \\(c_i \sim \texttt{Categorical}(\frac{1}{K}, \frac{1}{K}, ... \frac{1}{K}); \forall i=1..n\\)
- Cluster mean vector \\(\mu_k \in \mathbb{R}^d\\)
- \\(x_i | c_i ~ \mathcal{N}(\mu_k, I_d)\\)




$$
c_i \sim q(c_i; \omega_i)
$$
- $c = c_{1:N}$
- $\mu=\mu_{1:K}$
- $x=x_{1:N}$

$$
\begin{equation}
\begin{aligned}
	q(c_i; \omega_i) & \propto \exp \bigg\{
		\mathbb{E}_{q/q(c_i; \omega_i)}\big[
			\log p(\mu, c, x)
		\big]
	\bigg\}
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
& p(\mu, c, x) = p(\mu) \prod_{i=1}^N{ p(c_i) p(x_i | c_i, \mu)} \\
\iff & \log p(\mu, c, x) = \log p(\mu) + \sum_{i=1}^N \big[
	\log p(c_i) + \log p(x_i|c_i, \mu)
\big]

\end{aligned}
\end{equation}
$$



$$
\begin{equation}
\begin{aligned}
 & p(x_i | c_i, \mu) = \prod_{k=1}^K p(x_i |\mu_k)^{c_{ik}} \\ \implies & \log p(x_i | c_i, \mu) = \sum_{k=1}^K{c_{ik}\log p(x_i | \mu_k)}
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
& p(x_i|\mu_k) = \mathcal{N}(x_i; \mu_k, \mathbf{I}_d) = (2\pi)^{-d/2} \exp \{ -\frac{1}{2} (x_i - \mu_k)^\top(x_i - \mu_k) \} \\
\implies &
	\log p(x_i | \mu_k) = -\frac{1}{2} (x_i - \mu_k)^\top(x_i -\mu_k) + const
\end{aligned}
\end{equation}
$$



$$
\begin{equation}
\begin{aligned}
q(c_i; \omega_i) &\propto  \exp \bigg\{
	\mathbb{E}_{q / q(c_i; \omega_i)}\bigg[
		\sum_{j=1}^N \sum_{k=1}^Kc_{jk} \log p(x_j | \mu_k)
	\bigg]
\bigg\} \\
& = \exp\bigg\{ 
	\mathbb{E}_{q / q(c_i; \omega_i)}\bigg[
		\sum_{j\neq i} \sum_{k=1}^Kc_{jk} \log p(x_j | \mu_k) + \sum_{k=1}^Kc_{ik} \log p(x_i | \mu_k)
	\bigg]
\bigg\}\\
& = \exp \bigg\{ 
	\mathbb{E}_{q / q(c_i; \omega_i)}\bigg[ 
		\sum_{k=1}^K c_{ik}\log p(x_i | \mu_k) 
	\bigg]+ const
\bigg\} \\
& \propto \exp \bigg\{
\sum_{k=1}^K c_{ik} {\mathbb{E}_{\mu_k}\big[
	\log p(x_i |\mu_k)
\big]}
\bigg\} \\
& \propto \exp \bigg\{
	\sum_{k=1}^K c_{ik} \mathbb{E}_{\mu_k} \bigg[
		-\frac{1}{2}(x_i - \mu_k)^\top (x_i - \mu_k); m_k, s_k^2
	\bigg]
\bigg\} \\ 
& = \prod_{k=1}^K \exp\{ {\mathbb{E}_{\mu_k}[-\frac{1}{2}(x_i - \mu_k)^\top(x_i -\mu_k)]} \}^{c_{ik}} \\

\implies \omega_{ik} &= \exp \{\mathbb{E}_{\mu_k}[-\frac{1}{2}(x_i - \mu_k)^\top(x_i -\mu_k); m_k, s^2_k] \}
\end{aligned}
\end{equation} 
$$



$$
\begin{equation}
\begin{aligned}
	\mathbb{E}_{\mu_k}[-\frac{1}{2}(x_i - \mu_k)^\top(x_i -\mu_k); m_k, s^2_k] &= 
		-\frac{1}{2} \mathbb{E}_{\mu_k}[x_i^\top x_i - 2 x_i^\top\mu_k + \mu_k^\top\mu_k] \\
		&= -\frac{1}{2} \{
			x_i^\top x_i - 2 x_i^\top m_k + \mathbb{E}_{\mu_k}[\mu_k^\top \mu_k]
		\} \\
		& = -\frac{1}{2}\{x_i^\top x_i - 2x_i^\top m_k + m_k^\top m_k  + \texttt{tr}(s^2_k) \}
\end{aligned}
\end{equation}
$$

$$
\implies \omega_{ik} \propto \exp [x_i^\top m_k - \frac{1}{2} m_k^\top m_k - \frac{1}{2}\texttt{tr}(s^2_k)] 
$$


## Deriving the coordinate update for $\mu_k$
$$
\begin{equation}
\begin{aligned}
	x=1
\end{aligned}
\end{equation}
$$
-->
