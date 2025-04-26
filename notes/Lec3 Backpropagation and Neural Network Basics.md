# Do some math!
- Named Entity Recognition (NER)
#### Setup:
$$
x\to h=f\left( Wx+b \right) \to s=u^Th\to\sigma \left( s \right) =\frac{1}{1+e^{-s}}
$$
We will ignore the logistics step to simplify the math
## Computing Gradients by Hand
$$
\frac{{\partial \mathbf{h}}}{\partial \mathbf{z}}=\begin{pmatrix}
f' \left( z_{1} \right) &&\\
&\ddots&\\
&&f'\left( z_{n} \right) 
\end{pmatrix}=\text{diag}\left( \boldsymbol{f}'\left( z \right) 
 \right) 
$$
$$
\frac{{\partial}}{\partial \boldsymbol{x} }\left( \boldsymbol {W}  \boldsymbol {x}+\boldsymbol {b}  \right)=\boldsymbol {W} 
$$
$$
\frac{{\partial}}{\partial \boldsymbol{b} }\left( \boldsymbol {W}  \boldsymbol {x}+\boldsymbol {b}  \right)=\boldsymbol {I} 
$$
$$
\frac{ \partial  }{ \partial x } \left( \boldsymbol {u}^T\boldsymbol {h}   \right) =\boldsymbol{h}^T
$$
### Part 1
$$\frac{ \partial s }{ \partial b }=u^T\circ f'(z)=\delta$$
$\delta$ is the local error signal, which is a row vector
### Part 2
$$
\frac{ \partial s }{ \partial W } =\delta \frac{ \partial z }{ \partial W }  =\begin{bmatrix}
\frac{ \partial s }{ \partial W_{11}} &\cdots& \frac{ \partial s }{ \partial W_{1m}}\\
\vdots&\ddots&\vdots\\
\frac{ \partial s }{ \partial W_{11}}&\cdots &\frac{ \partial s }{ \partial W_{11}}
\end{bmatrix}=\delta^Tx^T
$$
> $\frac{ \partial s }{ \partial W_{{ij}} }=\delta_{i}x_{j}$
### What shape should the output be?
1. Jacobian: $\frac{ \partial s }{ \partial b }$ is a row vector
2. Shape convention: $\frac{ \partial s }{ \partial b }$ is a column vector (which makes the implementing SGD easier)
# Backpropagation
![image.png](https://i.imgur.com/N0Co7Rc.png)

![image.png](https://i.imgur.com/gjOirYs.png)

One mail goal is to reduce duplicated computation