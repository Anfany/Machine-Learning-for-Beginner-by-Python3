# Boosting 介绍

+ **Boosting思想**

   1. 每一次都根据上一次训练得到的模型结果，调整数据集样本分布，然后再生成下一个模型；
   2. 直到生成Ｍ个模型;
   3. 根据Ｍ个模型的结果集成得到最终的结果；
  
+ **集成方式**

   每个模型的重要度作为每个模型结果的权重，然后加权计算得出结果。
   
 可以看出Boosting中生成多个模型的方式并不是和Bagging一样并行生成，而是串行生成，因此也决定了多个模型结果的集成是**串行集成**,也就是每个模型的结果权重并不是一样的。如何来调整样本分布以及计算模型的重要度，不同方法有不同的定义，详情参见具体方法。
 
+ **代表方法**

   + **[AdaBoost](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/tree/master/Boosting/AdaBoost)**
   + **[GBDT](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/tree/master/Boosting/GBDT)**
   + **[XGBoost](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/tree/master/Boosting/XGBoost)**


