# 汽车价格预测
汽车价格预测是机器学习线性回归模拟中最常见的一种项目，我们基于数据集训练模型，并用于对给定特征值的汽车的价格进行预测      

---    
本项目中，我编写了两个python脚本来实现汽车价格的线性回归模拟:    
1. *carparice-predict* 脚本中，使用传统数学方法进行建模、计算rmse（使用`numpy` 、`pandas`包）      
2. *car_price_with_sklearn* 脚本中:       
   * 使用`sklearn.metrics`包中的`mean_squared_error`来计算rmse————均方误差     
   * 使用`sklearn.model_selection`包中的`train_test_split`来分割数据集     
   * 使用`sklearn.feature_extraction`包中的`DivVectorizer`来处理分类变量和转化为数组      
   * 使用`sklearn.linear_model`包中的`Ridge`来进行*岭回归*拟合     
   &emsp; &emsp; &emsp; &emsp;&emsp; &emsp; &emsp; &emsp;  &emsp; &emsp;&emsp; &emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; *岭回归*————含正则化的线性回归










