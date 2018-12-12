# LightGBM程序文件


  + **回归问题：北京市pm2.5预测**
  
     + 数据处理：[pm25_LightGBM_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/pm25_LightGBM_Data.py)
     
     + 模型建立：[LightGBM_Regression_pm25.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/LightGBM_Regression_pm25.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/lightgbm_pm25.jpg) 
  
        * 预测真实值与输出值对比曲线 
     
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/duibi_lightgbm.jpg)
         
 
  
  + **分类问题：成年人收入**
    
     + 数据处理：[adult_LightGBM_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/adult_LightGBM_Data.py)
     
     + 模型建立：[LightGBM_Classify_adult.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/LightGBM_Classify_adult.py)
     
     + 结果图示
     
         * 方法选择
       
           ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Boosting/LightGBM/lightgbm_adult.jpg) 
  
        * 预测数据集结果
        
           * 混淆矩阵
   
           |  混淆矩阵 | 预测<=50K | 预测>50K |
           |:-------|:-------|:-------|
           | 实际<=50K |   11676|   759    |
           |  实际>50K |    1316 |   2530  |

           
           * F1度量、精确率、召回率
           
           F1度量：0.833827117032927, 精确率：0.872550826116332, 召回率：7983944593006881
