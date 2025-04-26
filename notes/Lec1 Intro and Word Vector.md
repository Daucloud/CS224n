- Distributional semantics: A word's meaning is given by the words that frequently appear close-by
- Word vectors are also called word embeddings and word representations
## Word2Vec
- a framework for learning word vectors
![image.png](https://img.picgo.net/2025/02/10/202502102334472856d16506c03eea3ae.png)
### loss function 
- also known as objective function or cost function
Given position $t=1,\dots,T$ , window of a fixed size $m$ and center word $w_{j}$
- Likelihood
$$
L(\theta)=\prod_{t=1}^{T}\prod_{-m\leq j\leq m, j\neq 0}P\left( w_{t+j}\mid w_{t};\theta \right) 
$$
- Loss function:
$$
J(\theta)=-\frac{1}{T}\ln L\left( \theta \right) 
$$
Minimizing objective function equals to maximizing predictive accuracy
### How to cal $P\left( w_{t+j}\mid w_{t};\theta \right)$
1.  given two vectors to each word $w$
	- $v_{w}$ when $w$ is a center word
	- $u_{w}$ when $w$ is a context word
$$
P\left( o\mid c \right) =\frac{\exp(u_{o}^Tv_{c})}{\sum_{w\in V}\exp \left( u_{w}^Tv_{c} \right) }
$$
> Just an example for softmax function
> max: amplifies probability of largest $x_{i}$
> soft: still assigns some probability to smaller $x_{i}$
### Train the model
optimize $\theta$
$$
\theta=\left[  \begin{matrix}
v_{w_{1}}\\
v_{w_{2}}\\
\vdots\\
v_{w_{V}}\\
u_{w_{1}}\\
u_{w_{2}}\\
\vdots\\
u_{w_{V}}\\
\end{matrix} \right]\in \mathbb{R}^{2dV}
$$
- $d$: $d$-dimensinal vectors
- $V$: $V$-many words
### gradient descendent algorithm
- optimize $\theta$ by walking down the gradient to find the smallest $J\left( \theta \right):$ 
let $j=\ln P(o\mid c)$, do simple differentiation then we get:
$$
\frac{{\partial j}}{\partial v_{c}}=u_{o}-\sum_{x\in V}P(x\mid c)u_{x}
$$
This is in the form of **observed - expected**, which is always the case for softmax-style functions
