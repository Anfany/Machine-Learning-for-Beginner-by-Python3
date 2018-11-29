# AdaBoost程序文件说明

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_RF_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/pm25_RF_Data.py)
     
     + 模型建立：[pm25_RF_Regression.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/pm25_RF_Regression.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/method.jpg) 
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/duibi.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_RF_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/adult_RF_Data.py)
     
     + 模型建立：[adult_RF_Classify.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/adult_RF_Classify.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Bagging/Random_Forest/method_adult1.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11627 |   808    |
           |  实际>50K |    1324 |   2522  |

           
           * F1度量、精确率、召回率
           
           F1度量：0.8305868207860023, 精确率：0.8690498126650698, 召回率：0.7953841724235917
           
