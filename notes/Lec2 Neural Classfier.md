## Gradient descent
### Basic Idea
$$
\theta_{new}=\theta_{old}-\alpha \nabla _{\theta}J(\theta)
$$
**Nobody uses it: it's too slow!**
### Stochastic gradient descent(SGD)
Update the parameter on a small batch.
> This is just the so-called batch size.

## Word2Vec: more details
1. Skip-grams(SG): center->context
2. Continuous Bag of Words (CBOW): context->center
### SGNS: Skip Gram Negative Sampling
- Motivation: The naive $P\left( o\mid c \right)$ is to expensive to compute, since we have to sum the whole word table to get the denominator
- $J_{t}\left( \theta \right)=\log\sigma \left( u_{o}^T v_{c}\right)+\sum_{i=1}^{k}\mathbb E_{j\sim P \left( w \right)}\left[ \log\sigma \left(- u_{j}^Tv_{c} \right) \right]$: the former represents the positive sampling and the latter represents the negative sampling
- $J \left( \theta \right)=\frac{1}{T}\sum_{t=1}^T J_{t}\left( \theta \right)$
- $\sigma \left( x \right)=\frac{1}{1+e^{-x}}$: map any real number to $\left( 0,1 \right)$
-  $P \left( w \right)=\frac{U\left( w \right)^{3/4}}{Z}$, which will lower the weight of the common words
## GloVe
### Loss
- $J=\sum_{i,j=1}^V f \left( X_{ij} \right)\cdot \left( w_{i}^T\tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{ij} \right)^2$
- $f\left( X_{ij} \right)=\begin{cases}\left( \frac{x}{x_{max}} \right)^\alpha&\text{if}~ x<x_{max}\\1& \text{if}~ x\geq x_{max}\end{cases}$: lower the influence of the common words
### Evaluation of word Vectors
#### Intrinsic
- On a specific/intermediate subtask
#### Extrinsic
- Real tasks

