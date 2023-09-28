---
title: Noise constrastive estimation
draft: false 
date: 2023-09-23
tags: [learning, probabilistic-ml]
---



Documenting my note why studying [**Noise-contrastive estimation** paper](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf). The paper proposed a method to estimate the probability density function of a dataset by discriminating observed data and noised draw from a some distribution.

## Notation

- $p_d(x)$ true probability density function (p.d.f) of data.
- $p_n(x)$ p.d.f of noise generating distribution.
- $r(x) = \frac{1}{1+\exp(-x)}$ sigmoid function.
- $X = (x_1, ... x_T); x \sim p_d(x)$ be the dataset of T observations.
- $Y = (y_1, ... y_T); y \sim p_n(y)$ be the dataset of T artificially generated noise.
- $p_m(.; \theta)$ is estimation of $p_d(.)$ parameterized by $\theta$.

The goal


## Threorem I

> $\tilde{J}$ attains a maximum at $f(.) = \log p_d(.)$. There are no other extrema if the noise density $p_n(.)$ is chosen such it is nonzero whenever $p_d(.)$ is nonzero.

\begin{equation}
\begin{aligned}
\tilde{J}(\theta) = \frac{1}{2}\mathbb{E}_{x, y} {
    \log{r\big(f(x) - \log{p_n(x)}\big)} 
    + \log{\big[
        1 - r\big(f(y) - \log{p_n(y)}\big)
    \big]}
}
\end{aligned}
\end{equation}

### Proof

Let $\hat{f}(x)$ be the optimal function that maximizes $\tilde{J}$, and $f(x)=\hat{f}(x) + \epsilon\eta(x)$.

\begin{equation}
\begin{aligned}
    \tilde{J}(\theta) &= K(\epsilon) \\\
    &= \frac{1}{2}\mathbb{E}_{x, y} {
    \log{
        r\big(f(x) - \log{p_n(x)}\big)
    } + \log{\big[
        1 - r\big(f(y) - \log{p_n(y)}\big)
            \big]}
    } \\\
    &= \frac{1}{2}\underbrace{
        \mathbb{E}_x {
            \log r\big(
                    f(x) - \log p_n(x)
                    \big) 
        }
    }_A + 
    \frac{1}{2} \underbrace{\mathbb{E}_y {
        \log{\big[
            1 - r\big(f(y) - \log{p_n(y)}\big)
            \big]}
    }}_B \\\
    \implies \frac{dK}{d\epsilon} &= \frac{dA}{d\epsilon} + \frac{dB}{d\epsilon}
\end{aligned}
\end{equation}

Expand the first term of $K(\epsilon)$

\begin{equation}
\begin{aligned}
    A(\epsilon) &=  \mathbb{E}_x {
        \log r\big(
                f(x) - \log p_n(x)
                \big) 
    } \\\
    & = \int_x {
        p_d(x) \log{
            r\big(
                \hat{f}(x) + \epsilon \eta(x) - \log p_n(x)
            \big)
        } dx
    } 
\end{aligned}
\end{equation}

Taking derivative of $A(\epsilon)$

\begin{equation}
\begin{aligned}
    \frac{dA}{d\epsilon} &= \frac{1}{d\epsilon} \int_x {
        p_d(x) \log{
            r\big(
                \hat{f}(x) + \epsilon \eta(x) - \log p_n(x)
            \big)
        } dx
    } \\\
    & = \int_x {
        p_d(x) \big[ 
            \frac{1}{d\epsilon}\log{
                r \big(
                        \underbrace{
                            \hat{f}(x) + \epsilon \eta(x) - \log p_n(x)
                        }_{g(\epsilon)}
                \big)
            }
        \big]dx
    } \\\
    & = \int_x{
        p_d(x)
        \frac{d\log{r}}{dr}
        \frac{dr}{dg}
        \frac{dg}{d\epsilon}
        dx
    } \\\
    & = \int_x{
        p_d(x)
        \frac{1}{r}
        r(1-r)
        \eta(x)
        dx
    } \\\
    & = \int_x{
        p_d(x)
        \big[1 - r\big( \hat{f}(x) + \epsilon \eta(x) - \log p_n(x)\big) \big]
        \eta(x) dx
    }
\end{aligned}
\end{equation}

Now let's turn our attention to the second term of $K(\epsilon)$

\begin{equation}
\begin{aligned}
    B(\epsilon) &= \mathbb{E}_y {
        \log\big[
            1 - r\big(f(y) - \log{p_n(y)}\big)
        \big]
    } \\\
    & = \int_y {
        p_n(y)
        \log \big[
            1 - r \big(
                \underbrace{
                    \hat{f}(y) + \epsilon \eta(y) - \log p_n(y)
                }_h
            \big)
        \big]dy
    }
\end{aligned}
\end{equation}

Taking derivative of $B$ w.r.t $\epsilon$

\begin{equation}
    \begin{aligned}
        \frac{dB}{d\epsilon} &= \frac{1}{d\epsilon} \int_y{
            p_n(y)\log{
                \big[
                    1 - r\big( h(\epsilon)\big)
                \big]
            }dy
        } \\\
        &= \int_y {
            p_n(y)
            \frac{d\log(1-r)}{d(1-r)}
            \frac{d(1-r)}{dr}
            \frac{dr}{dh}
            \frac{dh}{d\epsilon}
            dy
        } \\\
        & = \int_y {
            p_n(y)
            \frac{1}{1-r}
            (-1)
            r(1-r)
            \eta(y)
        } \\\
        & = -\int_y{
            p_n(y) 
            r\big(
                    \hat{f}(y) + \epsilon \eta(y) - \log p_n(y)
            \big) 
            \eta(y) dy
        }
\end{aligned}
\end{equation}

Substitute result from eq(4) and eq(6) to eq(2), $\frac{dK}{d\epsilon}$ is evaluated to $0$ at $\epsilon = 0$.

\begin{equation}
\begin{aligned}
    \frac{dK}{d\epsilon}\big\vert_{\epsilon=0} 
        &= \frac{dA}{d\epsilon}\big\vert_{\epsilon=0}
        + \frac{dB}{d\epsilon}\big\vert_{\epsilon=0} \\\
        &= \int_x {
            p_d(x)
                \big[1 - r\big( \hat{f}(x) - \log p_n(x)\big) \big]
                \eta(x) dx 
        } \\\
        & - \int_y{
            p_n(y) 
            r \big(
                    \hat{f}(y) - \log p_n(y)
            \big) 
            \eta(y) dy 
        } \\\ 
        & = 0
\end{aligned}
\end{equation}

Consider eq. (7), if the support for $x$ and $y$ are equal, which mean we integrate $x$ and $y$ over a same region, we can change $y$ to $x$ and rewrite eq.(7) as

\begin{equation}
\begin{aligned}
    \frac{dK}{d\epsilon} \big\vert_{\epsilon = 0}
        &= \int_x {
            \underbrace{
                p_d(x)
                \big[1 - r\big( \hat{f}(x) - \log p_n(x)\big) \big]
            }_C
            \eta(x) dx 
        } \\\
        & - \int_x{
            \underbrace{
                p_n(x) 
                    r \big(
                        \hat{f}(x) - \log p_n(x)
                    \big) 
            }_D
            \eta(x) dx 
        } \\\
        & = \int_x{(C-D)\eta(x)dx} = 0 \quad \forall \eta(x)
\end{aligned}
\end{equation}

The equality in eq.(8) happend if and only if $C=D$. This result easily leads to $\hat{f}(x) = \log p_d(x)$.


## References
1. [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

