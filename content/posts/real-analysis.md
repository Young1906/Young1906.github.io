---
title: Real Analysis - Lecture notes
draft: false 
date: 2024-2-19
tags: [learning, real-analysis]
---


Notes I took during studying MIT OCW [Real Analysis](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The class taught by Professor Casey Rodriguez, he also taught [Functional analysis](https://tarheels.live/crodriguez/168-2/).

## Resources (Useful link)
- [Video lecture](https://www.youtube.com/watch?v=LY7YmuDbuW0&list=PLUl4u3cNGP61O7HkcF7UImpM0cR_L2gSw)
- [Course's homepage](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/)

## Lecture notes

Goal of the course
	- Gain  experience with proofs
	- Prove statements about the real numbers, function and limits

### Lecture 1: Sets, Set operations, and Mathematical Induction

**Definition** (Sets) A sets is a collection of objects called elements/members.

**Definition** (Empty set) A set with no elements, denoted as \\\(\emptyset\\\)

**Notation**
- \\\(a \in S\\\): \\\(a\\\) is a element of \\\(S\\\)
- \\\(a \notin S\\\): \\\(a\\\) is not a element of \\\(S\\\)
- \\\(\forall\\\): for all
- \\\(\exists\\\): there exists
- \\\(\implies\\\): implies
- \\\(\iff\\\): if and only if

**Definition** 
1) (subset) A set \\\(A\\\) is a subset of \\\(B\\\), denoted as \\\(A \subset B\\\) if: \\\(a \in A \implies a \in B\\\)
2) (equal) Two sets are equal if \\\(A \subset B \land B \subset A\\\)
3) (proper subset) \\\(A \subsetneqq B \iff A \subset B \land A \neq B\\\)


Set building notation

$$
    \\{ x \in A : P(x) \\}
$$

**Examples**

1) \\\(\mathbb{N} = \\{1, 2, 3 \cdots\\}\\\)
2) \\\(\mathbb{Z} = \\{\cdots,-2, -1, 0, 1, 2, \cdots\\}\\\)
3) \\\(\mathbb{Q} = \\{\frac{m}{n}: m, n \in \mathbb{Z}\\}\\\)
4) Real number set \\\(\mathbb{R}\\\)

**Remark**: \\\(\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}\\\)

**Goal**: describe the real number set $\mathbb{R}$

**Definition** (union) The union of A and B is the set
$$
A \cup B := \{x: x\in A \lor x\in B\}
$$
**Definition** (intersection) The intersection of A and B is the set
$$
A \cap B := \{x: x\in A \land x \in B\}
$$
**Definition** (different) The set different between A w.r.t B is the set 
$$
A\backslash B = \{x\in A: x\notin B\}
$$
**Definition** (complement) A complement of set A is the set 
$$
A^c = \{x: x\notin A\}
$$
**Definition** (disjoint) Two sets A and B are disjoint if \\\(A \cap B = \emptyset\\\).

**Theorem** (De-Morgan) If A, B, C are sets then 
1) \\\((B \cup C)^c = B^c \cap C^c\\\)
2) \\\((B \cap C)^c = B^c \cup C^c\\\)
3) \\\(A\backslash (B\cup C) = (A\backslash B) \cap (A\backslash C)\\\)
4) \\\(A\backslash (B\cap C) = (A\backslash B) \cup(A\backslash C)\\\)

**Induction** A way to prove theorem about natural number.

\\\(\mathbb{N} = \\{1, 2, 3, \cdots \\}\\\) has an ordering \\\(1 < 2 < 3< 4 < \cdots\\\)

**Axiom** (Well ordering of natural numbers) if \\\(S\subset \mathbb{N}\\\) and \\\(S\neq \emptyset\\\) has a least element \\\(\exists x\in S\\\) st \\\(\forall y \in S: x\leq y\\\).

**Theorem** (Induction) Let \\\(P(n)\\\) be a statement depending on \\\(n\in \mathbb{N}\\\). Assume:

1) (Base case) \\\(P(1)\\\) is true
2) (Inductive step) If \\\(P(m)\\\) is true, then \\\(P(m+1)\\\) is true.
Then \\\(P(n)\\\) is true for all \\\(n\in \mathbb{N}\\\).

**Proof**: Let \\\(S = \{n\in\mathbb{N}: P(n) \text{ is not true}\}\\\).
**Want to show** \\\(S=\emptyset\\\)

- Suppose \\\(S\neq \emptyset\\\). By WOP.\\\(\mathbb{N}\\\), \\\(S\\\) has a least element \\\(x\in S\\\). Since \\\(P(1)\\\) is true, \\\(1\notin S\\\), so \\\(x>1\\\).

- Since \\\(x\\\) is the least element in \\\(S\\\) \\\(\implies x-1 \notin S\\\).

By the definition of \\\(S\\\), \\\(P(x-1)\\\) is true, by 2) \\\(\implies P(x)\\\) is true \\\(\implies x \notin S\\\).

\\\(\therefore S = \emptyset\\\)

**Using induction** We want to prove some statement \\\(\forall n\in\mathbb{N}:P(n)\\\) is true, we have to do two things:
1) Prove \\\(P(1)\\\).
2) Prove \\\(P(m) \implies P(m+1)\\\)

**Example** For all \\\(c\neq 1, \forall n\in\mathbb{N}\\\):

$$
1 + c + c^2 + \cdots +c^n=\frac{1-c^{n+1}}{1-c}
$$

**Proof**

1) (Base case):

$$
1 + c^1 = \frac{1-c^{1+1}}{1-c}=\frac{(1-c)(1+c)}{1-c} = {1+c}; \forall c\neq1
$$

2) (Inductive step) Assume:
$$
	1+c+c^2+\cdots+c^m=\frac{1-c^{m+1}}{1-c}\quad(*)
$$
We want to show
$$
	1+c+c^2+\cdots+c^n=\frac{1-c^{n+1}}{1-c}\quad(**)
$$
for \\\(n = m+1\\\).

We have:

$$
\begin{aligned}
1+c+c^2+\cdots+c^m+c^{m+1} &= \frac{1-c^{m+1}}{1-c}+c^{m+1} \\\
& = \frac{1-c^{m+1}+c^{m+1}-c^{m+2}}{1-c}\\\
& = \frac{1-c^{(m+1)+1}}{1-c}
\end{aligned}
$$
So (\*) hold for \\\(n=m+1\\\). By induction, \\\(P(n)\\\) is true \\\(\forall n\in\mathbb{N}\\\).

