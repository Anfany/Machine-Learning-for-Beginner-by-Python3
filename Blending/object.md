# Blending 实例程序

  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_Stacking_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/pm25_Stacking_Data.py)
     
     + Stacking第二层模型建立：[BP_Regression.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/BP_Regression.py)
     
    + 最终模型建立：[Stacking_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_Regression_pm25.py)
     
     + 结果图示
     
         * 第一层各个模型结果
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_yiceng.jpg) 
           
         * 第二层BPNN训练成本曲线
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_two.jpg)            
           
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_last.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_Stacking_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/adult_Stacking_Data.py)
     
     + Stacking第二层模型建立：[bp_Classify.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/bp_Classify.py) 
     
     + 最终模型建立：[Stacking_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_Classify_adult.py)
     
     + 结果图示
     
         * 第一层各个模型结果
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Stacking/Stacking_adult8.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11640|   795   |
           |  实际>50K |    1292 |   2554  |
