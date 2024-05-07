---
title: Expectation Maximization - EM
date : 2024-04-15
tags : [learn, ml]
draft: False 
categories: [
    "Machine Learning",
    ]
---

## Problem 

Given a statistical model \\(P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta}) = P(\boldsymbol{X} | \boldsymbol{Z}, \boldsymbol{\theta})\\), which generate set of observations \\(\boldsymbol{X}\\), where \\(\boldsymbol{Z}\\) is a latent variable and unknow parameter vector \\(\boldsymbol{\theta}\\). The goal is to find \\(\boldsymbol{\theta}\\) that maximize the marginal likelihood:

$$
\mathcal{L}(\boldsymbol{\theta}; \boldsymbol{X}) = P(\boldsymbol{X} | \boldsymbol{\theta})
= \int_{\boldsymbol{Z}}P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta})d\boldsymbol{Z}
$$

As an example for this type of problem, there are two (unfair) coin A and B with probability of head for each coin is \\(p_A(H) = p \text{ and } p_B(H) = q\\). For each trial, we select coin A with probability \\(p(A) = \tau\\) and coin B with probability \\(p(B) = 1 -\tau\\), toss the coin and record the observation. The set of observations \\(\boldsymbol{X}\\) is the record of head or tail \\(\\{H, T, H, H, \cdots\\}\\), the latent variable which is unobserved is which coint is selected for each trail \\(\\{A, B, B, A, \cdots\\}\\), and the unknown parameter vector \\(\boldsymbol{\theta} = [p, q, \tau]\\). The goal is to find \\(\boldsymbol{\theta}\\) that best fit observations; EM is an instance of Maximum Likelihood Estimation (MLE).


## The EM algorithm
The EM algorithm seeks for \\(\boldsymbol{\theta}\\) by first initiates a random parameter vector \\(\boldsymbol{\theta}^{(0)}\\) and then iteratively performs two steps, namely the expectation step (E step) and the maximization step (M step): 

- (The E step) the expected loglikelihood of \\(\boldsymbol{\theta}\\), with respect to the current conditional distribution of \\(\boldsymbol{Z}\\) given observations \\(\boldsymbol{X}\\) and current estimation of \\(\boldsymbol{\theta}^{(t)}\\)

$$
Q(\boldsymbol{\theta} | \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\boldsymbol{Z} \sim P(. | \boldsymbol{X}, \boldsymbol{\theta}^{(t)})} {[
    \log P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta})
    ]}
$$

- (The M step) update parameter vector \\(\boldsymbol{\theta}\\)

$$
\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta} | \boldsymbol{\theta}^{(t)})
$$


### EM for the coin example

**Setup**
- Parameter vector \\(\boldsymbol{\theta} = [p, q, \tau]\\), and its estimation at step (t) is \\(\boldsymbol{\theta}^{(t)} = [p_t, q_t, \tau_t]\\)
- The \\(i^{th}\\) observation \\(x^{(i)}\\) is either head (H) or tail (T).
- The coin selected for the \\(i^{th}\\) trail \\(z^{(i)}\\) is either A or B:
    - \\(p(z^{(i)} = A) = \tau\\) 
    - \\(p(z^{(i)} = B) = 1 -\tau\\).

    For both cases, 
    $$
    \begin{equation}
    p(z^{(i)}) = \tau^{\mathbb{I}(z^{(i)}=A)}(1-\tau)^{\mathbb{I}(z^{(i)}=B)}
    \end{equation}
    $$

- When selected the coin A,
    - Probability that we get a head (H): \\(p(x^{(i)}=H | z^{(i)} = A) = p\\)
    - Probability that we get a head (T): \\(p(x^{(i)}=T | z^{(i)} = A) = 1 - p\\)

    For both cases,
    $$
    \begin{equation}
    p(x^{(i)} | z^{(i)}=A) = p^{\mathbb{I}(x^{(i)}=H)}(1 - p)^{\mathbb{I}(x^{(i)}=T)}
    \end{equation}
    $$

- Similarly, when B is selected
    $$
    \begin{equation}
    p(x^{(i)} | z^{(i)}=B) = q^{\mathbb{I}(x^{(i)}=H)}(1 - q)^{\mathbb{I}(x^{(i)}=T)}
    \end{equation}
    $$

Where \\(\mathbb{I}(\cdot)\\) is an indicator function on a predicate
    $$
    \mathbb{I}(p) = \begin{cases}
        1 \quad \text{if } p \text{ is True}\\\
        0 \quad \text{otherwise}
    \end{cases}
    $$


Once again, we generalize for both cases of \\(z^{(i)}\\)

$$
\begin{equation}
\begin{aligned}
p(x^{(i)} | z^{(i)}) = 
    [p^{\mathbb{I}(x^{(i)}=H)}(1 - p)^{\mathbb{I}(x^{(i)}=T)}]^{\mathbb{I}(z^{(i)}=A)}\\\
    \times [q^{\mathbb{I}(x^{(i)}=H)}(1 - q)^{\mathbb{I}(x^{(i)}=T)}]^{\mathbb{I}(z^{(i)}=B)}
\end{aligned}
\end{equation}
$$

The equation looks rather ugly, we can simplify this by encoding head as 1 and tail as 0; coin A as 1 and coin B as 0. The equation above can be written as

$$
\begin{equation}
p(x^{(i)} | z^{(i)}) = [p^{x^{(i)}}(1-p)^{1 - x^{(i)}}]^{z^{(i)}} 
[q^{x^{(i)}}(1-q)^{1 - x^{(i)}}]^{1-z^{(i)}}
\end{equation}
$$

Similarly for \\(p(z^{(i)})\\)
$$
\begin{equation}
    p(z^{(i)}) = \tau^{z^{(i)}}(1-\tau)^{1-z^{(i)}}
\end{equation}
$$

**Applying EM algorithm**

- **The (E step)**:
    - Construct the joint likelihood of a single pair of observation and latent variable \\(p(x^{(i)}, z^{(i)})\\ | \boldsymbol{\theta})\\). For the conciseness, we drop the \\((i)\\) superscript from the equation.
    
    $$
    \begin{equation}
    \begin{aligned}
        p(x, z | \boldsymbol{\theta}) = & p(x | z, \boldsymbol{\theta})p(z | \boldsymbol{\theta})\\\
        = & [p^{x}(1-p)^{1 - x}]^{z} [q^{x}(1-q)^{1 - x}]^{1-z} \tau^{z}(1-\tau)^{1-z}
        & \text{\tiny(from eq. 5 and 6)}
    \end{aligned}
    \end{equation}
    $$

    - Likelihood over entire observations \\(\boldsymbol{X}\\) and latent \\(\boldsymbol{Z}\\): 

        $$\boldsymbol{X}\odot\boldsymbol{Z} := \\{(x^{(i)}, z^{(i)})\\}_{i=1\cdots N}$$

        > A side note is that I am not entirely sure that \\(\odot\\) operator is appropriate in this situation.

        $$
        \begin{equation}
            \begin{aligned}
                P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta}) =& \prod_{(x, z) \in \boldsymbol{X}\odot\boldsymbol{Z}} {
                   p(x, z | \boldsymbol{\theta}) 
                }
            \end{aligned}
        \end{equation}
        $$

    - Log likelihood of the joint probability

        $$
        \begin{equation}
        \begin{aligned}
        \log P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta}) & = \sum_{(x, z)} \log p(x, z | \boldsymbol{\theta})
        \end{aligned}
        \end{equation}
        $$

        > Taking a log always seem to make thing to be better.

    - Finally, we need to take the expectation of the log likelihood w.r.t conditional probability of \\(\boldsymbol{Z}|\boldsymbol{X}, \boldsymbol{\theta}^{(t)}\\)
        - Posterior for a single latent \\(z\\)

            $$
                \begin{equation}
                \begin{aligned}
                p(z | x, \boldsymbol{\theta}^{(t)})
                & = \frac{p(x, z | \boldsymbol{\theta}^{(t)})}
                    {p(x | \boldsymbol{\theta}^{(t)})} & \text{\tiny(Bayes Theorem)}\\\
                & = \frac{p(x, z | \boldsymbol{\theta}^{(t)})}
                    {
                        p(x, z = 0| \boldsymbol{\theta}^{(t)}) +
                        p(x, z = 1| \boldsymbol{\theta}^{(t)})
                    } & \text{\tiny(Marginal likelihood over z in denominator)}\\\
                & = \frac{
                    [p_t^{x}(1-p_t)^{1 - x}]^{z} [q_t^{x}(1-q_t)^{1 - x}]^{1-z} \tau_t^{z}(1-\tau_t)^{1-z}
                }{
                    q_t^{x}(1-q_t)^{1 - x} (1-\tau_t) + p_t^{x}(1-p_t)^{1 - x}\tau_t
                } & \text{\tiny(from eq. 7)}
                \end{aligned}
                \end{equation}
            $$

        - Taking the expectation

            $$
            \begin{equation}
            \begin{aligned}
                Q(\boldsymbol{\theta} | \boldsymbol{\theta}^{(t)}) 
                    &= \mathbb{E}_{\boldsymbol{Z} | \boldsymbol{X}, \boldsymbol{\theta}^{(t)}}{\bigg[
                        \sum\_{(x, z)}{\log p(x, z | \boldsymbol{\theta})}
                    \bigg]} \\\
                    &= \sum\_{(x, z)} {
                       \mathbb{E}\_{\boldsymbol{Z} | \boldsymbol{X}, \boldsymbol{\theta}^{(t)}}{[
                            \log p(x, z | \boldsymbol{\theta})
                       ]}
                    } \\\
                    &= \sum\_{(x, z)} {
                       \mathbb{E}\_{z | x, \boldsymbol{\theta}^{(t)}}{[
                            \log p(x, z | \boldsymbol{\theta})
                       ]}
                    } \\\
            \end{aligned}
            \end{equation}
            $$

            > It is always bothering for me that in literature, the posterior, of which to be taken expectation over, for the entire set latent variables \\(\boldsymbol{Z} = \\{ z^{(1)}, \cdots z^{(n)}\\}\\) can be replaced by the posterior for a single latent \\(z\\) in (eq. 11) without explanation. So in order to understand this, consider the equation.            

            $$
                \begin{aligned}
                \mathbb{E}_{\boldsymbol{Z}}{\bigg[\sum\_{z\in \boldsymbol{Z}}{f(z)}\bigg]} &= \int\_{\boldsymbol{Z}}{
                    \bigg[\sum\_{z\in\boldsymbol{Z}} f(z)\bigg] p(\boldsymbol{Z}) d\boldsymbol{Z}
                } \\\
                & = \sum\_{z\in\boldsymbol{Z}}{\int\_{\boldsymbol{Z}}{f(z)}} p(\boldsymbol{Z})d\boldsymbol{Z} \\\
                & = \sum\_{z\in\boldsymbol{Z}}{
                   \int\_{\boldsymbol{Z}\text{/}z}
                        \underbrace{\bigg[\int\_{z}f(z)p(z)dz\bigg]}\_{A=\mathbb{E}\_z[f(z)]}
                    p(\boldsymbol{Z}\text{/}z)d(\boldsymbol{Z}/z)
                } \\\
                & = \sum\_{z\in\boldsymbol{Z}} A 
                    \int\_{\boldsymbol{Z}\text{/}z} p(\boldsymbol{Z}\text{/}z)d(\boldsymbol{Z}/z) & \text{\tiny(A is constant w.r.t variable being integrated over)} \\\
                & = \sum\_{z\in\boldsymbol{Z}} \mathbb{E}_z[f(z)] & \text{\tiny(Integeral overal a p.d.f evalulated to 1)}
                \end{aligned}
            $$

            Where \\(\boldsymbol{Z} = \\{z^i\\}_{i=1\cdots N}; z \sim p(Z)\\); \\(\boldsymbol{Z}/z\\) denotes set all variables within \\(\boldsymbol{Z}\\) except \\(z\\).

        - Having clear that up, we are able to resume from (eq. 11)
            $$
            \begin{aligned}
                Q(\boldsymbol{\theta} | \boldsymbol{\theta}^{(t)}) 
                    &= \sum\_{(x, z)} {
                       \mathbb{E}\_{z | x, \boldsymbol{\theta}^{(t)}}{[
                            \log p(x, z | \boldsymbol{\theta})
                       ]}
                    } \\\
                    &= \sum\_{(x, z)} {\bigg[
                        p(z = 0 | x, \boldsymbol{\theta}^{(t)}) \log p(x, z = 0 | \boldsymbol{\theta}) \\
                        + p(z = 1 | x, \boldsymbol{\theta}^{(t)}) \log p(x, z = 1 | \boldsymbol{\theta})
                    \\bigg]}
            \end{aligned}
            $$

            From (eq. 7)

            - 
                $$ 
                \begin{aligned}
                p(x, z = 0 |\boldsymbol{\theta}) = q^x(1-q)^{1-x}(1-\tau)
                \end{aligned}
                $$
            - 
                $$ 
                \begin{aligned}
                p(x, z = 1 |\boldsymbol{\theta}) = p^x(1-p)^{1-x}\tau
                \end{aligned}
                $$

            From (eq. 10)

            - 
                $$ 
                \begin{aligned}
                p(z = 0 | x, \boldsymbol{\theta}^{(t)})
                & = \frac{
                    q_t^{x}(1-q_t)^{1-x}(1-\tau_t)
                }{
                    q_t^{x}(1-q_t)^{1 - x} (1-\tau_t) + p_t^{x}(1-p_t)^{1 - x}\tau_t
                }
                \end{aligned}
                $$

            - 
                $$ 
                \begin{aligned}
                p(z = 1 | x, \boldsymbol{\theta}^{(t)})
                & = \frac{
                    p_t^{x}(1-p_t)^{1-x}\tau_t
                }{
                    q_t^{x}(1-q_t)^{1 - x} (1-\tau_t) + p_t^{x}(1-p_t)^{1 - x}\tau_t
                }
                \end{aligned}
                $$


## Proof of correctness
T.B.D

## EM for Gaussian Mixture Model

- [EM for GMM's python implementation](https://github.com/young1906/em)

---
**Foot note**
- I am preparing for my graduate school application, this post is written in preparation for the application and interview.
- I suppose to be preparing a slide for my supervisor, but sometime you can't help when the mood strikes. So sorry in advance to my supervisor.
- Lately I haven't been myself due to the stress of application process, so writing this helps keeping me on track, somehow.
