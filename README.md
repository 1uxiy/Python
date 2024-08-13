# python-机器学习
<font face="仿宋"> 关于python机器学习的练习 </font>  
机器学习的步骤（CRISP-DM）：
问题理解 $\longrightarrow$ 数据理解 $\longrightarrow$ 数据准备 $\longrightarrow$ 建模 $\longrightarrow$ 评估 $\longrightarrow$部署 $\longrightarrow$ 迭代

 &ensp;     
                
## 线性回归机器学习 

1. 利用*标准方程*计算权重向量  
   $w = (X^{T}X)^{-1}X^{T}y$            &ensp; &ensp; &ensp; &ensp; &ensp; &ensp; **X并非由dataframe直接转化而来，需要在最前面加上一列1，用于区分偏置项和其他项**
<table><tr><td bgcolor=cyan> $w = (X^{T}X+\alpha I)^{-1}X^{T}y$ &ensp; ($I$为单位矩阵) </td></tr></table>
     
2. 利用*权重向量*进行预测  
   $g(x_{i}) = w_{0} + \sum_{n=1}^{\infty}(x_{in}*w_{n}) = w_{0} + x_{i} \cdot w$ 
    
3. *RMSE*:评估模型质量  
   $RMSE = \sqrt{\frac{1}{m}{\sum_{i=1}^{m}(g(X_{i})-y_{i})^2}}$
  
&ensp; 以上三点为线性回归机器学习建模的数学原理支撑,还有一些较为重要的细节：
1. **分类变量的处理**：  
   例如，在汽车价格预测的模型中，存在分类变量车们的个数：2、3、4，对于诸如此类的分类变量，我们无法直接用标准方程来计算,所以，我们选择的解决办法是，将单个分类变量转化为多个特征值，并用布尔值（0、1）来计算标准方程
   比如，将car_doors这个分类变量转化为car_doors_2,car_doors_3,car_doors_4三个特征值
   ```
   def class_feature(df,name,n,vector):
    global features_1
    features_1 = vector.copy()

    rank=df[name].value_counts()
    if len(rank.index)<n:
        for value in rank.index:
            feature = '%s_%s'%(name,value)
            features_1.append(feature)
            df[feature] = (df[name]==value).astype(int)
    else:
        for value in rank.index[:n]:
            feature = '%s_%s'%(name,value)
            features_1.append(feature)
            df[feature] = (df[name]==value).astype(int)
   ```  
3. **数据的正则化**：   
   正则化是为了防止过度拟合，即模型在训练数据上表现良好，但在新的数据上表现很差，因为模型学习到了数据的噪声  
   · 一种正则化的方式是在数据矩阵的每个对角元素上添加一个小数值  
   这种情况下标准方程可以修正为:   
&ensp; <table><tr><td bgcolor=cyan> $w = (X^{T}X+\alpha I)^{-1}X^{T}y$ &ensp; ($I$为单位矩阵) </td></tr></table>
         
&ensp;     
&ensp;        
                   
## 用于分类的机器学习  
 目标变量是一种分类变量
   
1. **风险率** ： 分组比率与总体比率之间的比值   
   &ensp;  &ensp; ***风险=分组率/总体率***    
   &emsp; 风险是一个介于0到无穷的数字，反应分组中的元素与总体相比有多大可能产生影响    
   &emsp; 风险接近1时，说明分组与其余人群具有相同的风险水平  
2. **查看所有分类变量的风险率和差值**：         
   &emsp; 利用循环查看所有分类变量的风险率和差值，可以提前确定特征的重要性————这帮助我们回答了“究竟是什么在影响目标变量？”           
  ```python
  for col in categorical:   
      df_group = df_train_full.groupby(col).churn.agg(['mean'])     
      df_group['diff'] = df_group['mean'] - global_mean       
      df_group['risk'] = df_group['mean'] / global_mean     
      display(df_group)
```
  
   &emsp; **请注意display和print的区别**    
   &emsp; 但重要性指标只能帮助我们衡量分类变量和目标变量的依赖关系，很难用它来说明最重要的特征是什么     
3. **互信息**：       
   &emsp; 互信息是衡量分类变量和目标变量依赖程度的重要指标  
   &emsp; 互信息值越高，依赖程度越高，分类变量越重要   
   &emsp; `Scikit-learn`已经在`metrics`包的`mutual_info_score`函数中实现了互信息的计算    
   &emsp; `mutual_info_score(series1,series2)`
