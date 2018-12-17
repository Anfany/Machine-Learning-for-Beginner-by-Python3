# CatBoost 程序文件说明

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_CatBoost_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/pm25_CatBoost_Data.py)
     
     + 模型建立：[CatBoost_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/CatBoost_Regression_pm25.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/CatBoost_pm25.jpg) 
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/duibi_CatBoost.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_CatBoost_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/adult_CatBoost_Data.py
     
     + 模型建立：[CatBoost_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/CatBoost_Classify_adult.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/CatBoost/CatBoost_adult.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11706 |   729    |
           |  实际>50K |    1342 |   2504  |

           
           * F1度量、精确率、召回率
           
           F1度量：0.8327518701740704, 精确率：0.8727965112708065, 召回率：0.7962205967128915
