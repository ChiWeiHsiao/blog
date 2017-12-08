---
title: 深度學習--Backpropagation的推導
date: 2016-10-07 20:22:42
tags:
---
# Backpropagation
> forward propagation 可參考 [UFLDL](http://deeplearning.stanford.edu/wiki/index.php/Neural_Networks)

$ABC$
$$A\_D$$
更新 weight 使得 cost function 變小： $W_{ji}^{(l)} := W_{ji}^{(l)} - \eta \nabla_{W_{ji}^{(l)}}J(W)$

所以要找到 $$\nabla_{W_{ji}^{(l)}}J(W) \Rightarrow$$

=====================