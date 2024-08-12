# python-机器学习
<font face="仿宋"> 关于python机器学习的练习 </font>  
机器学习的步骤（CRISP-DM）：
问题理解 $\longrightarrow$ 数据理解 $\longrightarrow$ 数据准备 $\longrightarrow$ 建模 $\longrightarrow$ 评估 $\longrightarrow$部署 $\longrightarrow$ 迭代

## 线性回归机器学习 

1. 利用*标准方程*计算权重向量  
   $w = (X^{T}X)^{-1}X^{T}y$
     
2. 利用*权重向量*进行预测  
   $g(x_{i}) = w_{0} + \sum_{n=1}^{\infty}(x_{in}*w_{n}) = w_{0} + x_{i} \cdot w$ 
    
3. *RMSE*:评估模型质量  
   $RMSE = \sqrt{\frac{1}{m}{\sum_{i=1}^{m}(g(X_{i})-y_{i})^2}}$
  
以上三点为线性回归机器学习建模的数学原理支撑
还有一些较为重要的细节：
1. **分类变量的处理**：
   例如，在汽车价格预测的模型中，存在分类变量车们的个数：2、3、4，对于诸如此类的分类变量，我们无法直接用标准方程来计算
   所以，我们选择的解决办法是，将单个分类变量转化为多个特征值，并用布尔值（0、1）来计算标准方程
  
2. **数据的正则化**：
   正则化是为了防止过度拟合，即模型在训练数据上表现良好，但在新的数据上表现很差，因为模型学习到了数据的噪声
   · 一种正则化的方式是在数据矩阵的每个对角元素上添加一个小数值  
   这种情况下标准方程可以修正为:  
      $w = (X^{T}X+\alpha I)^{-1}X^{T}y$

