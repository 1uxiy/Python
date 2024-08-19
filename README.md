# python-机器学习
<font face="仿宋"> 关于python机器学习的练习 </font>  
机器学习的步骤（CRISP-DM）：
问题理解 $\longrightarrow$ 数据理解 $\longrightarrow$ 数据准备 $\longrightarrow$ 建模 $\longrightarrow$ 评估 $\longrightarrow$部署 $\longrightarrow$ 迭代

 &ensp;     
                
## 线性回归机器学习 

1. 利用*标准方程*计算权重向量  
   &emsp; $w = (X^{T}X)^{-1}X^{T}y$       
 **X并非由dataframe直接转化而来，需要在最前面加上一列1，用于区分偏置项和其他项**     
&emsp; <table><tr><td bgcolor=cyan> $w = (X^{T}X+\alpha I)^{-1}X^{T}y$ &ensp; ($I$为单位矩阵) </td></tr></table>
     
2. 利用*权重向量*进行预测  
   &emsp; $g(x_{i}) = w_{0} + \sum_{n=1}^{\infty}(x_{in}*w_{n}) = w_{0} + x_{i} \cdot w$ 
    
3.  *RMSE*:评估模型质量  
   &emsp; $RMSE = \sqrt{\frac{1}{m}{\sum_{i=1}^{m}(g(X_{i})-y_{i})^2}}$
  
&ensp; 以上三点为线性回归机器学习建模的数学原理支撑,还有一些较为重要的细节：
1. **分类变量的处理**：  
   &emsp; 例如，在汽车价格预测的模型中，存在分类变量车们的个数：2、3、4，对于诸如此类的分类变量，我们无法直接用标准方程来计算,所以，我们选择的解决办法是，将单个分类变量转化为多个特征值，并用布尔值（0、1）来计算标准方程
   比如，将car_doors这个分类变量转化为car_doors_2,car_doors_3,car_doors_4三个特征值
   ```python
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
   &emsp; 正则化是为了防止过度拟合，即模型在训练数据上表现良好，但在新的数据上表现很差，因为模型学习到了数据的噪声  
   &emsp; · 一种正则化的方式是在数据矩阵的每个对角元素上添加一个小数值  
   &emsp; 这种情况下标准方程可以修正为:   
&ensp; &emsp;  <table><tr><td bgcolor=cyan> $w = (X^{T}X+\alpha I)^{-1}X^{T}y$ &ensp; ($I$为单位矩阵) </td></tr></table>              
   &emsp;             
***线性回归可以直接使用`sklearn.linear_model`包中的的`LinearRegression`，正则化回归则可以使用同一个包中的`Ridge`***     
         
&ensp;     
&ensp;        
                   
## 用于分类的机器学习  
 目标变量是一种分类变量
   
1. **风险率** ： 分组比率与总体比率之间的比值   
   &ensp;  &ensp; ***风险=分组率/总体率***    
   &emsp; 风险是一个介于0到无穷的数字，反应分组中的元素与总体相比有多大可能产生影响    
   &emsp; 风险接近1时，说明分组与其余人群具有相同的风险水平  
 &emsp;  
2. **查看所有分类变量的风险率和差值**：         
   &emsp; 利用循环查看所有分类变量的风险率和差值，可以提前确定特征的重要性————这帮助我们回答了“究竟是什么在影响目标变量？”     
   &emsp; 但重要性指标只能帮助我们衡量分类变量和目标变量的依赖关系，很难用它来说明最重要的特征是什么                 
  ```python
  for col in categorical:   
      df_group = df_train_full.groupby(col).churn.agg(['mean'])     
      df_group['diff'] = df_group['mean'] - global_mean       
      df_group['risk'] = df_group['mean'] / global_mean     
      display(df_group)  #请注意display和print的区别       
```    
3. **互信息**：       
   &emsp; 互信息是衡量分类变量和目标变量依赖程度的重要指标  
   &emsp; 互信息值越高，依赖程度越高，分类变量越重要   
   &emsp; `Scikit-learn`已经在`metrics`包的`mutual_info_score`函数中实现了互信息的计算    
   &emsp; `mutual_info_score(series1,series2)`    
   &emsp; 互信息虽然能量化依赖程度，但他只能反映两个分类变量之间的依赖程度，对于数值变量就不行    
  &emsp;            
4. **相关系数**:
   &emsp; 正相关、负相关、零相关    
   &emsp; 相关系数的计算非常简单，使用`pd.corrwith(series1,series2)`即可    
   &emsp;          
5. **分类变量的独热编码**:        
   &emsp; 对于分类变量contact有几个可能的值（月签、年签、两年签），则年签合同可以按照独热编码表示为（0,1,0)         
   &emsp; 以此来将分类变量转化为可计算的数值      
 &emsp;   
6. **逻辑回归**：   
   &emsp; 逻辑回归也是线性模型，但与线性模型不同的是，它是一个分类模型，因为目标变量是二元的   
   &emsp; 逻辑回归输出的是**概率**，即 $y_i=1$的概率,因此我们要将他的输出控制在0~1之内        
   &emsp; 完整公式如下：    
   &emsp; &emsp;  $g(x_i) = sigmoid(w_0 +x^{T}_{i}w)$    
   &emsp; &emsp;  $sigmoid(x)={1 \over 1+exp(-x)}$       
   <img src="https://github.com/1uxiy/Python/blob/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/IMAGE/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20240814170553.jpg" width="50%" height="50%">     
&emsp;        
7. **模型**：    
   &emsp; 在这里，我们使用 `sklearn`库中`linear_model`模块中的`LogisticRegresssion`函数   
   &emsp; 并用`model.fit`来训练它     
   &emsp; 使用这种方法得到的模型，偏置项 $w_0$ 存储在`model.intercept_[0]`,权重向量 $w$ 存储在`model.coef_[0]`        
   &emsp; 对于逻辑回归，我们通常使用`predict_proba()`函数来预测目标变量，它返回一个二维数组，第一列表示目标为负(0)的概率，第二列表示目标为正(1)的概率   
     * 这种预测返回一个介于0和1之间的概率，我们称之为*软预测*    
     * 但是在实际中，我们需要做出是否向客户做额外推销的决定，即*硬预测*————返回值为二进制（0或1),我们只需要设置一个阈值，当概率大于这个值时归为1或0即可         
     * 独热编码的权重为正，则说明相应的特征指向目标变量为1，并且绝对值越大说明该特征的影响更大           
&emsp;      
8. **评估指标**:            
     * *准确度* ： 即正确预测案例(硬预测)的数量 &esnp; / &ensp; 数据集案例的总数，可以直接调用`sklearn.metrics`包中的`accurancy_score(y1,y2)`来计算                     
     * *虚拟基线*：实际应用中，仅通过准确度很难确定模型的质量。例如在一个总数为一千的数据集中，有七百个未流失、三百个流失，那么预测值如果全部为未流失，准确率都有70%。所以我们建立一个全部为`false`或者全部为`true`的数组来作为基线，从而为预测数值准确度的计算提供一个校准           
&emsp; 
9. **混淆矩阵**:
    &emsp; 将数据分为四类：       
     * TN:真实为F，预测也为F       
     * FN:真实为T，预测为F          
     * TP:真实为T，预测也为T      
     * FP:真实为F，预测为T        &emsp; &emsp; ——由这四个部分组成的矩阵即混淆矩阵。       
   &emsp; $准确度 = \frac{TN+TP}{TN+FN+TP+FP}$                 
   &emsp; $查准率 = \frac{TP}{TP+FP}$  ——预测为TRUE的客户中真正为TURE的比例            
   &emsp; $查全率 = \frac{TP}{TP+FN}$  ——实际为TRUE的客户中预测为TRUE的比例    
```python
t = 0.5
predict_churn = (y_pre >= t)
predict_no_churn = (y_pre < t)
actual_churn = (y_val == 1)
actual_no_churn = (y_val == 0)

TP = (predict_chrun & actual_churn).sum()             #布尔数组上调用sum（）返回的是TRUE的数量      
FP = (predict_chrun & actual_no_churn).sum()          #sum先将布尔数组转化为0、1，再求和  
TN = (predict_no_chrun & actual_no_churn).sum()
FN = (predict_no_chrun & actual_churn).sum()
```                
10. **ROC曲线和AUC分数**       
      * *随机基线模型*:数组中所有数值均为0~1之间的随即分数      
      * *理想模型*:理想模型以预测概率对数据集中的所有样本进行排序（其对流失客户的打分更高，所以预测概率更高）排序类似000000111           
   &emsp; &emsp; &emsp; 理想模型的预测是从0递增到1               
      * *FPR* :假正例率——实际为假的目标中预测为真的概率  &emsp; $FPR = /frac{FP}{FP+TN}$      
      * *TPR* :真正例率——实际为真的目标中预测为真的概率  &emsp; $TPR = /frac{TP}{TP+FN}$                
    &emsp; `ROC曲线`是以`FPR`为x轴、`TPD`为y轴绘制的曲线，通常将随机模型、理想模型和训练模型的曲线一起绘制、比对，越靠近理想模型，说明训练模型质量越高        






















