# Decision Tree Theory

决策树主要包括ID3，C4.5以及CART。下面给出三种算法的说明：

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Decision%20Tree/decisiontree.png)

C4.5是ID3的改进版，而CART是最常用的，因此本文主要介绍CART。

### CART

首先看下面表格中的示例数据(随机生成，仅供参考)。其中类似年龄，身高，月收入为连续变量，学历，工作为离散变量。

   + 如果把**动心**视为目标变量，此问题为**分类问题**。
   
   + 如果把**动心度**视为目标变量，此问题为**回归问题**。

|编号|年龄|学历|工作|月收入(k)|身高(cm)|动心|动心度|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0001|24|专科|国企|5|175|**否**|0.56|
|0002|35|博士|国企|12|184|**否**|0.49|
|0003|27|硕士|私企|13|179|**是**|0.76|
|0004|23|硕士|私企|6|182|**是**|0.67|
|0005|30|硕士|国企|8|167|**否**|0.58|
|0006|22|硕士|国企|10|166|**是**|0.60|
|0007|28|博士|国企|6|171|**是**|0.73|
|0008|38|博士|私企|23|183|**否**|0.40|
|0009|23|专科|私企|13|176|**否**|0.52|
|0010|30|博士|国企|11|180|**是**|0.88|

CART的目的是生成一个类似下面这样的树：分类树或者回归树。

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Decision%20Tree/TreCart.png)

叶子节点为是或者否，是分类树。是数字，为回归树。下面分别讲述回归树和分类树的生成方式：

* **分类树**

    ID3算法使用**信息增益**来选择特征，信息增益大的优先选择，这种方式会使得特征值较多的特征容易被选择【*示例数据中的学历可能会比工作优先选择，因为学历有3个值，而工作有2个值*】。
    
    在C4.5算法中，采用了**信息增益比**来选择特征，改进了ID3容易选择特征值多的特征的问题。C4.5也是优先选择较大的。
    
    上述2者都是基于信息论的**熵**模型的，这里面会涉及对数运算，因此计算成本较大。
    
    CART分类树算法使用**基尼系数**，既减少了计算成本，又保留了熵这种运算形式的优点。基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。

     + 基尼系数
     
     + 计算示例
     
         + 离散变量
         
         + 连续变量
         
 * **回归树**

      + 说明示例
      
         以上面给出的示例数据为例，下面说明2种形式的变量生成回归树的方式。将数据集合定义为**D**，
     
         + 离散变量
         
         以学历为例：
         
         1. 将数据集合D分为**Dsp1=D(学历=专科)**以及**Dsp2=D(学历!=专科)**，其中Dsp1中动心度构成的集合为**Msp1**，均值为**asp1**；Dsp2中动心度构成的集合为**Msp2**，均值为**asp2**；计算2个数据子集合的误差平方和的和值：
         
         <a href="http://www.codecogs.com/eqnedit.php?latex={\color{Red}&space;MSE(sp)=\sum_{di\in&space;Msp1}(di-asp1)^{2}&space;&plus;&space;\sum_{di\in&space;Msp2}(di-asp2)^{2}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?{\color{Red}&space;MSE(sp)=\sum_{di\in&space;Msp1}(di-asp1)^{2}&space;&plus;&space;\sum_{di\in&space;Msp2}(di-asp2)^{2}}" title="{\color{Red} MSE(sp)=\sum_{di\in Msp1}(di-asp1)^{2} + \sum_{di\in Msp2}(di-asp2)^{2}}" /></a>
        
         类似于MSE(sp)，遍历所有学历中的值，得到下面的MSE(ms)，MSE(dr)。
        
        <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Red}&space;MSE(ms)=\sum_{di\in&space;Mms1}(di-ams1)^{2}&space;&plus;&space;\sum_{di\in&space;Mms2}(di-ams2)^{2}}&space;\\&space;\\&space;\\&space;{\color{Red}&space;MSE(dr)=\sum_{di\in&space;Mdr1}(di-adr1)^{2}&space;&plus;&space;\sum_{di\in&space;Mdr2}(di-adr2)^{2}}&space;\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Red}&space;MSE(ms)=\sum_{di\in&space;Mms1}(di-ams1)^{2}&space;&plus;&space;\sum_{di\in&space;Mms2}(di-ams2)^{2}}&space;\\&space;\\&space;\\&space;{\color{Red}&space;MSE(dr)=\sum_{di\in&space;Mdr1}(di-adr1)^{2}&space;&plus;&space;\sum_{di\in&space;Mdr2}(di-adr2)^{2}}&space;\\" title="\\ {\color{Red} MSE(ms)=\sum_{di\in Mms1}(di-ams1)^{2} + \sum_{di\in Mms2}(di-ams2)^{2}} \\ \\ \\ {\color{Red} MSE(dr)=\sum_{di\in Mdr1}(di-adr1)^{2} + \sum_{di\in Mdr2}(di-adr2)^{2}} \\" /></a>
        
        其中
        
        数据集**Dms1=D(学历=硕士)** 以及 **Dms2=D(学历!=硕士)**，Dms1中动心度构成的集合为**Mms1**，均值为**ams1**；Dms2中动心度构成的集合为**Mms2**，均值为**ams2**；
        
        **Ddr1=D(学历=博士)** 以及 **Ddr2=D(学历!=博士)**，Ddr1中动心度构成的集合为**Mdr1**，均值为**adr1**；Ddr2中动心度构成的集合为**Mdr2**，均值为**adr2**
        
        + 计算示例
           
           <a href="http://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;\\\mathbf{Msp1}=[0.56,&space;0.52]\\&space;\\\boldsymbol{Msp2}=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88]\\&space;\\\mathbf{asp1}=0.54,&space;\mathbf{asp2}=0.63875\\&space;\\\mathbf{MSE(sp)}=[(0.56-0.54)^{2}&space;&plus;&space;(0.52-0.54)^{2}]\\&space;\\&space;&plus;&space;[(0.49-0.63875)^{2}&plus;\cdots&space;&plus;&space;(0.88-0.63875)^{2}]\\&space;\\=0.008&plus;0.1662875=0.1670875" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bg_white&space;\fn_phv&space;\\\mathbf{Msp1}=[0.56,&space;0.52]\\&space;\\\boldsymbol{Msp2}=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88]\\&space;\\\mathbf{asp1}=0.54,&space;\mathbf{asp2}=0.63875\\&space;\\\mathbf{MSE(sp)}=[(0.56-0.54)^{2}&space;&plus;&space;(0.52-0.54)^{2}]\\&space;\\&space;&plus;&space;[(0.49-0.63875)^{2}&plus;\cdots&space;&plus;&space;(0.88-0.63875)^{2}]\\&space;\\=0.008&plus;0.1662875=0.1670875" title="\\\mathbf{Msp1}=[0.56, 0.52]\\ \\\boldsymbol{Msp2}=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88]\\ \\\mathbf{asp1}=0.54, \mathbf{asp2}=0.63875\\ \\\mathbf{MSE(sp)}=[(0.56-0.54)^{2} + (0.52-0.54)^{2}]\\ \\ + [(0.49-0.63875)^{2}+\cdots + (0.88-0.63875)^{2}]\\ \\=0.008+0.1662875=0.1670875" /></a>
           
        我们可以得到
        
        得到以上结果后，在其中选择MSE最小的变量，如果最终的最佳特征为学历，则以此变量作为分割变量。
         
         
         + 连续变量


