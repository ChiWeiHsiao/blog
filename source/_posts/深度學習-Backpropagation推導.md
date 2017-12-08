---
title: 深度學習--Backpropagation的推導
mathjax: true
date: 2016-10-07 20:22:42
tags:
---
若想暸解 forward propagation 可參考 [UFLDL](http://deeplearning.stanford.edu/wiki/index.php/Neural_Networks)。

更新 weight 使得 cost function 變小： $W_{ji}^{(l)} := W_{ji}^{(l)} - \eta \nabla_{W_{ji}^{(l)}}J(W)$


所以要找到 $\nabla_{W_{ji}^{(l)}}J(W) \Rightarrow$

$$\begin{aligned}
    \nabla_{W_{ji}^{(l)}} J(W) &= \frac{\partial J(W)}{\partial W_{ji}^{(l)}} \\
    &= \frac{\partial J(W)}{\partial z_j^{(l+1)}} \color{Blue}{\frac{\partial z_j^{(l+1)}}{\partial W_{ji}^{(l)}}}\\
    & \because z_j^{(l+1)}=\sum_{\forall i}W_{ji}^{(l)} * \color{Blue}{a_i^{(l)}} \quad\therefore \frac{\partial z_j^{(l+1)}}{\partial W_{ji}^{(l)}} = \color{Blue}{a_i^{(l)}}  \\
    &=\color{Red}{\frac{\partial J(W)}{\partial z_j^{(l+1)}}} \color{Blue}{a_i^{(l)}} \qquad \\
    &= \color{Red}{ \delta_j^{(l+1)} }\color{Blue}{a_i^{(l)}}
\end{aligned}$$

所以，
$$\color{Red}{\frac{\partial J(W)}{\partial z_j^{(l+1)}} = \delta_j^{(l+1)} =\quad??}$$


$\because if  ~f(x)=s~is~sigmoid,~f^{'}()$

## error term $\delta^{(l)}_i$
$\delta^{(l)}_i$ 代表的是這個 node 對 Error (cost) 有多少責任
$$\frac{\partial J(W)}{\partial z_j^{(l+1)}} = \delta_j^{(l+1)}$$

### Example: 假設 cost 使用 least squared error
以下推導帶入 $J(W,b; x,y) = \frac{1}{2} \left\| h_{W,b}(x) - y \right\|^2$

#### output node 的 $\delta^{(L)}_i$
 $z_j^{(l+1)} \rightarrow [node] \rightarrow a_j^{(l+1)} \rightarrow Err = J(W)$
* 先考慮 output layer 只有單一 node
* 對 output node 來說 Error 直接 = J(W)
* 最後一層的層數: L 

$\begin{aligned}
\delta_j^{(L)} 
&= \frac{\partial J(W)}{\partial z_j^{(L)}} \\
&=\color{Blue}{ 
\frac{\partial J(W)}{\partial a_j^{(L)}}
}
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \\
&\because J(W,b; x,y) = \frac{1}{2} \left\| y - h_{W,b}(x) \right\|^2 = \frac{1}{2} \left\| y-a_j^{(L)} \right\|^2 \\
&= \color{Blue}{ 
\frac{
\partial \frac{1}{2} \left\| y - a_j^{(L)} \right\|^2}
{\partial a_j^{(L)}} 
}
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}  \\
&= \color{Blue}{
-(y - a_j^{(L)}) 
}
\color{Red}{
\frac{\partial a_j^{(L)}}{\partial z_j^{(L)}}
}\\
&= -(y - a_j^{(L)})
\color{Red}{
\frac{\partial ~f(z_j^{(L)})}{\partial z_j^{(L)}}
} \\
&= -(y - a_j^{(L)})~
\color{Red}{
f^{'}(z_j^{(L)})
} \\
&\qquad \color{Orange}{
\because a_j^{(L)} = sigmoid(z_j^{(L)}) }\\
&\qquad\color{Orange}{
\therefore \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} = sigmoid^{'}(z_j^{(L)}) = a_j^{(L)}[1-a_j^{(L)}] }\\
&= -(y - a_j^{(L)})~
\color{Red}{a_j^{(L)}[1-a_j^{(L)}]} \\
\end{aligned}$


#### hidden layer node 的 $\delta^{(l)}_i$
For $l = n_l-1, n_l-2, n_l-3, \ldots, 2$
For each node $i$ in layer $l$, set

$$\delta^{(l)}_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) f'(z^{(l)}_i)$$