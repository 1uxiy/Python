# python-机器学习
<font face="仿宋"> 关于python机器学习的练习 </font>  
机器学习的步骤（CRISP-DM）：
问题理解 $\longrightarrow$ 数据理解 $\longrightarrow$ 数据准备 $\longrightarrow$ 建模 $\longrightarrow$ 评估 $\longrightarrow$部署 $\longrightarrow$ 迭代

## 线性回归机器学习 

1. 利用标准方程计算权重向量  
   $w = (X^{T}X)^{-1}X^{T}y$
2. 利用权重向量进行预测  
   $g(x_{i}) = w_{0} + \sum_{n=1}^{\infty}x_{in}*w_{n}$
3. RMSE:评估模型质量  
   $RMSE = \sqrt{\frac{1}{m}{\sum_{i=1}^{m}(g(X_{i})-y_{i})^2}}$
