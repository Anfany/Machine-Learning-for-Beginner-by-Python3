# XGBoost程序文件

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_XGBoost_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/pm25_XGBoost_Data.py)
     
     + 模型建立：[XGBoost_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/XGBoost_Regression_pm25.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/xgboost_pm25.jpg) 
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/duibi_xgb.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_XGBoost_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/adult_XGBoost_Data.py)
     
     + 模型建立：[XGBoost_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/XGBoost_Classify_adult.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/XGBoost/xgboost_adult.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11664 |   771   |
           |  实际>50K |    1306 |   2540  |

           
           * F1度量、精确率、召回率
           
           F1度量：0.8342166034233478, 精确率：0.8724279835390947, 召回率：0.7992120022557235
