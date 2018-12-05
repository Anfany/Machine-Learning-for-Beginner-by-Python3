# GBDT程序文件

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_GBDT_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/pm25_GBDT_Data.py)
     
     + 模型建立：[GBDT_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/GBDT_Regression_pm25.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/AdaBoost/adaboost_pm25.jpg) 
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/AdaBoost/duibi.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_GBDT_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/adult_GBDT_Data.py)
     
     + 模型建立：[GBDT_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/GBDT_Classify_adult.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/GBDT/ada_adult1.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11654 |   781   |
           |  实际>50K |    1307 |   2539  |

           
           * F1度量、精确率、召回率
           
           F1度量：0.8336178642299452, 精确率：0.8717523493642897, 召回率：0.7986799061829783
