# Blending 实例程序

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_Blending_data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/pm25_Blending_data.py)
     
     + Blending第一层中的BPNN模型：[BP_Regression.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/BP_Regression.py)
     
     + Blending第二层线性回归模型：[Linear_Regression.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Linear_Regression.py)
     
     
    + 最终模型建立：[Blending_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_Regression_pm25.py)
     
     + 结果图示
     
         * 第一层各个模型结果
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_pm25.jpg) 
           
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_duibi_公式法.jpg)
         
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_duibi_梯度下降法.jpg) 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_Blending_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/adult_Blending_Data.py)
     
     + Blending第二层模型建立：[LR.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/LR.py) 
     
     + 最终模型建立：[Blending_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_Classify_adult.py)
     
     + 结果图示
     
         * 第一层各个模型结果
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_adult.jpg)
           
          * 第二层模型
       
            ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Blending/Blending_duibi.jpg)          
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11726|   709  |
           |  实际>50K |    1400 |   2447  |
