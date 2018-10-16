# Decision Tree Theory

决策树主要包括ID3，C4.5以及CART。下面给出三种算法的说明：

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Decision%20Tree/decisiontree.png)

C4.5是ID3的改进版，而CART是最常用的，因此本文主要介绍CART。

***

### CART

首先看下面表格中的示例数据(随机生成，仅供参考)。其中类似年龄，身高，月收入为连续变量，学历，工作为离散变量。

   + 如果把**动心**视为目标变量，此问题为**分类问题**。
   
   + 如果把**动心度**视为目标变量，此问题为**回归问题**。

|编号|年龄|学历|工作|月收入(k)|身高(cm)|动心|动心度|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0001|24|专科|国企|5|175|**N**|0.56|
|0002|35|博士|国企|13|180|**N**|0.49|
|0003|27|硕士|私企|13|173|**Y**|0.76|
|0004|23|硕士|私企|5|180|**N**|0.67|
|0005|30|硕士|国企|5|166|**N**|0.58|
|0006|22|硕士|国企|13|166|**Y**|0.60|
|0007|28|博士|国企|5|175|**Y**|0.73|
|0008|38|博士|私企|19|180|**N**|0.40|
|0009|23|专科|私企|13|175|**N**|0.52|
|0010|30|博士|国企|13|173|**Y**|0.88|

CART的目的是生成一个类似下面这样的树：分类树或者回归树。

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Decision%20Tree/CART_TREES.png)

叶子节点若为Y或者N，是分类树；若是数字，则为回归树。下面分别讲述回归树和分类树的生成方式：

* **分类树**

    ID3算法使用**信息增益**来选择特征，信息增益大的优先选择，这种方式会使得特征值较多的特征容易被选择【*示例数据中的学历可能会比工作优先选择，因为学历有3个值，而工作有2个值*】。
    
    在C4.5算法中，采用了**信息增益比**来选择特征，改进了ID3容易选择特征值多的特征的问题。C4.5也是优先选择较大的。
    
    上述2者都是基于信息论的**熵**模型的，这里面会涉及对数运算，因此计算成本较大。
    
    CART分类树算法使用**基尼系数**，既减少了计算成本，又保留了熵这种运算形式的优点。基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好。

     + **基尼系数**
     
       对于一个样本集合**S**，假设其包含**m**个不同的值，这**m**个值可看作**m**个不同的类。其中由**类i**组成的集合为**Si**，那么对于属于**类i**的样本点**k**而言，其概率为**P(k)=集合Si的样本个数除去集合S的样本个数**。则基于概率分布的基尼指数定义如下：
       
       <a href="http://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S)}&space;=&space;\mathbf{1}&space;-&space;\sum_{\mathrm{i}=1}^{\mathrm{m}}(\frac{|\mathbf{S_{i}}|}{|\mathbf{S}|})^{2}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S)}&space;=&space;\mathbf{1}&space;-&space;\sum_{\mathrm{i}=1}^{\mathrm{m}}(\frac{|\mathbf{S_{i}}|}{|\mathbf{S}|})^{2}}" title="{\color{Blue} \mathbf{G(S)} = \mathbf{1} - \sum_{\mathrm{i}=1}^{\mathrm{m}}(\frac{|\mathbf{S_{i}}|}{|\mathbf{S}|})^{2}}" /></a>
       
       其中符号||为计算集合内元素个数的符号，对于m等于2的情况，上面的式子等价与
       
       <a href="http://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S)}&space;=&space;\mathbf{2}\mathbf{P(k)}*(\mathbf{1}-\mathbf{P(k)})}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S)}&space;=&space;\mathbf{2}\mathbf{P(k)}*(\mathbf{1}-\mathbf{P(k)})}" title="{\color{Blue} \mathbf{G(S)} = \mathbf{2}\mathbf{P(k)}*(\mathbf{1}-\mathbf{P(k)})}" /></a>
       
       如果样本集合**S**，被某个规则R划分为n个数据子集，分别为S1， S2，……， Sn，则此时的计算基尼系数公示如下：
       
       <a href="http://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S,&space;R&space;)}&space;=&space;\sum_{\mathrm{\mathbf{j=1}}}^{\mathrm{\mathbf{n}}}\frac{\mathbf{|Sj|}}{\mathbf{|S|}}\mathbf{G(Sj)}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bg_white&space;\fn_phv&space;{\color{Blue}&space;\mathbf{G(S,&space;R&space;)}&space;=&space;\sum_{\mathrm{\mathbf{j=1}}}^{\mathrm{\mathbf{n}}}\frac{\mathbf{|Sj|}}{\mathbf{|S|}}\mathbf{G(Sj)}}" title="{\color{Blue} \mathbf{G(S, R )} = \sum_{\mathrm{\mathbf{j=1}}}^{\mathrm{\mathbf{n}}}\frac{\mathbf{|Sj|}}{\mathbf{|S|}}\mathbf{G(Sj)}}" /></a>
       
       在CART算法中，上述n的值一定为2。因为每一次分裂，都是把数据集合一分为二。
       
       针对离散、连续的变量，下面给出具体的计算基尼系数的步骤：
       
     + **离散变量：以工作为例**
     
        1. 将数据集合D分为**Dg1=D(学历=国企)**以及**Dg2=D(学历!=国企)**，其中Dg1中动心构成的集合为**Mg1**；Dg2中动心构成的集合为**Mg2**;
        
        此时的基尼系数计算如下：
        
        <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\boldsymbol{Mg1=[N,&space;N,&space;N,Y,Y,Y]}}\\&space;\\&space;{\color{Blue}&space;\boldsymbol{Mg2=[N,&space;N,N,Y]}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,gongzuo=guoqi)=\frac{|Mg1|}{|D|}G(Mg1)&plus;\frac{|Mg2|}{|D|}G(Mg2)}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{\frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]&plus;\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{0.45}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\boldsymbol{Mg1=[N,&space;N,&space;N,Y,Y,Y]}}\\&space;\\&space;{\color{Blue}&space;\boldsymbol{Mg2=[N,&space;N,N,Y]}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,gongzuo=guoqi)=\frac{|Mg1|}{|D|}G(Mg1)&plus;\frac{|Mg2|}{|D|}G(Mg2)}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{\frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]&plus;\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{0.45}}" title="\\ {\color{Blue} \boldsymbol{Mg1=[N, N, N,Y,Y,Y]}}\\ \\ {\color{Blue} \boldsymbol{Mg2=[N, N,N,Y]}}\\ \\ {\color{Blue} \mathbf{G(D,gongzuo=guoqi)=\frac{|Mg1|}{|D|}G(Mg1)+\frac{|Mg2|}{|D|}G(Mg2)}}\\ \\ \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots ={\color{Blue} \boldsymbol{\frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]+\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}]}}\\ \\ \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots ={\color{Blue} \boldsymbol{0.45}}" /></a>
        
        因为工作只有2个变量，因此按照工作为私企和工作为国企的基尼系数是相同的。
         
     + **连续变量：以月收入为例**
     
        1. 首先将月收入的值去重后，按照从小到大的排列顺序为[5, 13, 19],相邻的数值取平均值得到序列[9, 16]。类似于离散变量的情况，以9为例：将数据集分为Dn9=D(月收入<9)以及Dm9=D(月收入>9)，相应的动心组成的集合分别为**Mn9,Mm9**。
        
        下面给出计算基尼系数的过程：
        
        <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\boldsymbol{Mn9=[N,&space;N,&space;N,Y]}}\\&space;\\&space;{\color{Blue}&space;\boldsymbol{Mm9=[N,&space;Y,Y,N,N,Y]}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,yueshouru<>9)=\frac{|Mn9|}{|D|}G(Mn9)&plus;\frac{|Mm9|}{|D|}G(Mm9)}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\mathbf{\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}]&space;&plus;&space;\frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{0.45}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,yueshouru<>16)}}={\color{Blue}&space;\mathbf{\frac{9}{10}[1-(\frac{5}{9})^{2}-(\frac{4}{9})^{2}]&space;&plus;&space;\frac{1}{10}[1-(\frac{1}{1})^{2}-(\frac{0}{1})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\mathbf{0.44}}\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\boldsymbol{Mn9=[N,&space;N,&space;N,Y]}}\\&space;\\&space;{\color{Blue}&space;\boldsymbol{Mm9=[N,&space;Y,Y,N,N,Y]}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,yueshouru<>9)=\frac{|Mn9|}{|D|}G(Mn9)&plus;\frac{|Mm9|}{|D|}G(Mm9)}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\mathbf{\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}]&space;&plus;&space;\frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\boldsymbol{0.45}}\\&space;\\&space;{\color{Blue}&space;\mathbf{G(D,yueshouru<>16)}}={\color{Blue}&space;\mathbf{\frac{9}{10}[1-(\frac{5}{9})^{2}-(\frac{4}{9})^{2}]&space;&plus;&space;\frac{1}{10}[1-(\frac{1}{1})^{2}-(\frac{0}{1})^{2}]}}\\&space;\\&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;\cdots&space;={\color{Blue}&space;\mathbf{0.44}}\\" title="\\ {\color{Blue} \boldsymbol{Mn9=[N, N, N,Y]}}\\ \\ {\color{Blue} \boldsymbol{Mm9=[N, Y,Y,N,N,Y]}}\\ \\ {\color{Blue} \mathbf{G(D,yueshouru<>9)=\frac{|Mn9|}{|D|}G(Mn9)+\frac{|Mm9|}{|D|}G(Mm9)}}\\ \\ \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots ={\color{Blue} \mathbf{\frac{4}{10}[1-(\frac{3}{4})^{2}-(\frac{1}{4})^{2}] + \frac{6}{10}[1-(\frac{3}{6})^{2}-(\frac{3}{6})^{2}]}}\\ \\ \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots ={\color{Blue} \boldsymbol{0.45}}\\ \\ {\color{Blue} \mathbf{G(D,yueshouru<>16)}}={\color{Blue} \mathbf{\frac{9}{10}[1-(\frac{5}{9})^{2}-(\frac{4}{9})^{2}] + \frac{1}{10}[1-(\frac{1}{1})^{2}-(\frac{0}{1})^{2}]}}\\ \\ \cdots \cdots \cdots \cdots \cdots \cdots \cdots \cdots ={\color{Blue} \mathbf{0.44}}\\" /></a>
     
     年龄的处理方式相似，假如不考虑年龄，只考虑月收入和工作，则按着月收入为16这个界限分类，是最优的，因为0.44是最小值。
     
     
     + **树的输出**
     
     叶子节点数据集中，目标变量中占多数的类别，为这个叶子节点的输出类别。如果把上面给定的示例数据集看作一个叶子节点的话，如果某条数据正好落在这个数据集内，则这个数据的分类为N，因为这个数据集中有6条数据为N，多于Y的数据。
     
***
         
 * **回归树**
 
    以上面给出的示例数据为例，下面说明2种形式的变量生成回归树的方式。将数据集合定义为**D**，
     
     + **离散变量：以学历为例**
         
         1. 将数据集合D分为**Dsp1=D(学历=专科)**以及**Dsp2=D(学历!=专科)**，其中Dsp1中动心度构成的集合为**Msp1**，均值为**asp1**；Dsp2中动心度构成的集合为**Msp2**，均值为**asp2**；计算2个数据子集合的误差平方和的和值：
         
         <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\mathbf{MSE(sp)=\frac{|Msp1|}{|D|}\sqrt[2]{\sum_{di\in&space;Msp1}(di-asp1)^{2}}&plus;&space;\frac{|Msp2|}{|D|}\sqrt[2]{\sum_{di\in&space;Msp2}(di-asp2)^{2}}&space;}&space;}\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\mathbf{MSE(sp)=\frac{|Msp1|}{|D|}\sqrt[2]{\sum_{di\in&space;Msp1}(di-asp1)^{2}}&plus;&space;\frac{|Msp2|}{|D|}\sqrt[2]{\sum_{di\in&space;Msp2}(di-asp2)^{2}}&space;}&space;}\\" title="\\ {\color{Blue} \mathbf{MSE(sp)=\frac{|Msp1|}{|D|}\sqrt[2]{\sum_{di\in Msp1}(di-asp1)^{2}}+ \frac{|Msp2|}{|D|}\sqrt[2]{\sum_{di\in Msp2}(di-asp2)^{2}} } }\\" /></a>
        
         类似于MSE(sp)，遍历所有学历中的值，得到下面的MSE(ms)，MSE(dr)。
        
        <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\mathbf{MSE(ms)=\frac{|Mms1|}{|D|}\sqrt[2]{\sum_{di\in&space;Mms1}(di-ams1)^{2}}&plus;&space;\frac{|Mms2|}{|D|}\sqrt[2]{\sum_{di\in&space;Mms2}(di-ams2)^{2}}&space;}&space;}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(dr)=\frac{|Mdr1|}{|D|}\sqrt[2]{\sum_{di\in&space;Mdr1}(di-adr1)^{2}}&plus;&space;\frac{|Mdr2|}{|D|}\sqrt[2]{\sum_{di\in&space;Mdr2}(di-adr2)^{2}}&space;}&space;}\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\mathbf{MSE(ms)=\frac{|Mms1|}{|D|}\sqrt[2]{\sum_{di\in&space;Mms1}(di-ams1)^{2}}&plus;&space;\frac{|Mms2|}{|D|}\sqrt[2]{\sum_{di\in&space;Mms2}(di-ams2)^{2}}&space;}&space;}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(dr)=\frac{|Mdr1|}{|D|}\sqrt[2]{\sum_{di\in&space;Mdr1}(di-adr1)^{2}}&plus;&space;\frac{|Mdr2|}{|D|}\sqrt[2]{\sum_{di\in&space;Mdr2}(di-adr2)^{2}}&space;}&space;}\\" title="\\ {\color{Blue} \mathbf{MSE(ms)=\frac{|Mms1|}{|D|}\sqrt[2]{\sum_{di\in Mms1}(di-ams1)^{2}}+ \frac{|Mms2|}{|D|}\sqrt[2]{\sum_{di\in Mms2}(di-ams2)^{2}} } }\\ \\ {\color{Blue} \mathbf{MSE(dr)=\frac{|Mdr1|}{|D|}\sqrt[2]{\sum_{di\in Mdr1}(di-adr1)^{2}}+ \frac{|Mdr2|}{|D|}\sqrt[2]{\sum_{di\in Mdr2}(di-adr2)^{2}} } }\\" /></a>
        
        其中
        
        数据集**Dms1=D(学历=硕士)** 以及 **Dms2=D(学历!=硕士)**，Dms1中动心度构成的集合为**Mms1**，均值为**ams1**；Dms2中动心度构成的集合为**Mms2**，均值为**ams2**；
        
        **Ddr1=D(学历=博士)** 以及 **Ddr2=D(学历!=博士)**，Ddr1中动心度构成的集合为**Mdr1**，均值为**adr1**；Ddr2中动心度构成的集合为**Mdr2**，均值为**adr2**
        
        + **计算示例**
           
           <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\mathbf{Msp1=[0.56,0.52],}}&space;{\color{Blue}&space;\mathbf{asp1=0.54}}\\&space;\\&space;{\color{Blue}&space;\mathbf{Msp2=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88],}}&space;{\color{Blue}&space;\mathbf{asp2=0.63875}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(sp)=\frac{2}{10}&space;\sqrt[2]{([(0.56-0.54)^{2}&plus;(0.52-0.54)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots&space;&plus;&space;\frac{8}{10}&space;\sqrt[2]{([(0.49-0.63875)^{2}&plus;\cdots&space;&plus;(0.88-0.63875)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.005657&plus;&space;0.081557}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.087214}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(ms)=0.21404}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(dr)=0.229774}}\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\mathbf{Msp1=[0.56,0.52],}}&space;{\color{Blue}&space;\mathbf{asp1=0.54}}\\&space;\\&space;{\color{Blue}&space;\mathbf{Msp2=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88],}}&space;{\color{Blue}&space;\mathbf{asp2=0.63875}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(sp)=\frac{2}{10}&space;\sqrt[2]{([(0.56-0.54)^{2}&plus;(0.52-0.54)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots&space;&plus;&space;\frac{8}{10}&space;\sqrt[2]{([(0.49-0.63875)^{2}&plus;\cdots&space;&plus;(0.88-0.63875)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.005657&plus;&space;0.081557}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.087214}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(ms)=0.21404}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(dr)=0.229774}}\\" title="\\ {\color{Blue} \mathbf{Msp1=[0.56,0.52],}} {\color{Blue} \mathbf{asp1=0.54}}\\ \\ {\color{Blue} \mathbf{Msp2=[0.49,0.76,0.67,0.58,0.60,0.73,0.40,0.88],}} {\color{Blue} \mathbf{asp2=0.63875}}\\ \\ {\color{Blue} \mathbf{MSE(sp)=\frac{2}{10} \sqrt[2]{([(0.56-0.54)^{2}+(0.52-0.54)^{2}])}}}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots + \frac{8}{10} \sqrt[2]{([(0.49-0.63875)^{2}+\cdots +(0.88-0.63875)^{2}])}}}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots=0.005657+ 0.081557}}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots=0.087214}}\\ \\ {\color{Blue} \mathbf{MSE(ms)=0.21404}}\\ \\ {\color{Blue} \mathbf{MSE(dr)=0.229774}}\\" /></a>
  
        得到以上结果后，MSE最小的为**MSE(sp)** ，其对应的特征值为**专科**，可以说特征学历最小的MSE为MSE(sp)，如果最终的最佳特征为学历，则以学历是否为专科作为分类标准。
      
      + **连续变量：以身高为例**
      
          1. 将身高的所有值**去重**后按照**从小到大**的顺序排列，得到集合H=[166,173,175,180], 取**相邻两个数的中间值**得到集合MH=[169.5,174,177.5],接下来的计算就类似于离散变量的情况，挨个遍历，把数据分为小于、大于这2个数据子集。以169.5为例，把数据集分为**Dn169.5=D(身高<169.5)和Dm169.5=D(身高>169.5)**，数据子集相应的动心度组成的集合分别为**Mn169.5,Mm169.5**,集合相应的均值为**an169.5, am169.5**。
          
        174，177.5的符号命名规则类似，不一一赘述。
          
          + **计算示例**
          
          <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;{\color{Blue}&space;\mathbf{Mn169.5=[0.58,&space;0.60],}}&space;{\color{Blue}&space;\mathbf{an169.5=0.59}}\\&space;\\&space;{\color{Blue}&space;\mathbf{Mm169.5=[0.56,&space;0.49,&space;0.76,&space;0.67,&space;0.73,&space;0.40,&space;0.52,&space;0.88],}}&space;{\color{Blue}&space;\mathbf{am169.5=0.62625}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(169.5)=\frac{2}{10}&space;\sqrt[2]{([(0.58-0.59)^{2}&plus;(0.60-0.59)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots&space;&plus;&space;\frac{8}{10}&space;\sqrt[2]{([(0.56-0.62625)^{2}&plus;\cdots&space;&plus;(0.88-0.62625)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.002828&space;&plus;0.084944&space;}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.087772}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(174)=0.20636}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(177.5)=0.360627}}\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;{\color{Blue}&space;\mathbf{Mn169.5=[0.58,&space;0.60],}}&space;{\color{Blue}&space;\mathbf{an169.5=0.59}}\\&space;\\&space;{\color{Blue}&space;\mathbf{Mm169.5=[0.56,&space;0.49,&space;0.76,&space;0.67,&space;0.73,&space;0.40,&space;0.52,&space;0.88],}}&space;{\color{Blue}&space;\mathbf{am169.5=0.62625}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(169.5)=\frac{2}{10}&space;\sqrt[2]{([(0.58-0.59)^{2}&plus;(0.60-0.59)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots&space;&plus;&space;\frac{8}{10}&space;\sqrt[2]{([(0.56-0.62625)^{2}&plus;\cdots&space;&plus;(0.88-0.62625)^{2}])}}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.002828&space;&plus;0.084944&space;}}\\&space;\\\&space;{\color{Blue}&space;\mathbf{\cdots&space;\cdots&space;\cdots&space;\cdots=0.087772}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(174)=0.20636}}\\&space;\\&space;{\color{Blue}&space;\mathbf{MSE(177.5)=0.360627}}\\" title="\\ {\color{Blue} \mathbf{Mn169.5=[0.58, 0.60],}} {\color{Blue} \mathbf{an169.5=0.59}}\\ \\ {\color{Blue} \mathbf{Mm169.5=[0.56, 0.49, 0.76, 0.67, 0.73, 0.40, 0.52, 0.88],}} {\color{Blue} \mathbf{am169.5=0.62625}}\\ \\ {\color{Blue} \mathbf{MSE(169.5)=\frac{2}{10} \sqrt[2]{([(0.58-0.59)^{2}+(0.60-0.59)^{2}])}}}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots + \frac{8}{10} \sqrt[2]{([(0.56-0.62625)^{2}+\cdots +(0.88-0.62625)^{2}])}}}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots=0.002828 +0.084944 }}\\ \\\ {\color{Blue} \mathbf{\cdots \cdots \cdots \cdots=0.087772}}\\ \\ {\color{Blue} \mathbf{MSE(174)=0.20636}}\\ \\ {\color{Blue} \mathbf{MSE(177.5)=0.360627}}\\" /></a>
          
          可知对于特征身高最小的MSE为MSE(169.5)。如果最终的最佳特征是身高，则要把数据集分为身高高于169.5cm、以及不足169.cm的两部分数据子集。
          
     从以上可以看出身高是比学历还要好的分类特征，其他特征就不一一计算。到此，对离散、连续的变量的处理方式的说明已经结束。按照上述的方式就可以将分裂形成的数据集再次进行分裂，形成树。
     

   + **树的输出**
   
        树是通过一个个叶子节点决定输出的。输出的方式也包括2种：
      
       +  **回归树**：叶子节点代表的数据子集中目标变量的均值，就作为输出值。例如示例中的叶子节点，其输出值为0.56+0.52=0.54。当要预测的某条数据恰好属于这个数据子集，则针对这条数据的动心度的预测值就是0.54。
          
       +  **模型树**：对于一个叶子节点来说，有输入，也有对应的输出。 根据输入和输出的关系，建立模型，这个模型可以是线性回归，也可以通过神经网络来建立。这个叶子节点的输出值是所建立的模型的输出值。当要预测的某条数据恰好属于这个数据子集，则针对这条数据的动心度的预测值就是将数据带入建立的模型中得到的值。
       
***
        
+  **停止生长**
   
    上面说明了怎么生成一个树。当然树不能无限生长下去，这里说一下什么时候让某个数据子集停止分裂，也就是将这个数据子集形成一个叶子节点。一般有2种方式：
        
      + 通过限制树的深度。树的深度和人们计算家谱有几代方式相同，只不过比其少一代。示例图中显示出来的树的深度是3， 如果是家谱的就可看作4代。
        
      + 通过给定MSE或者基尼系数的阈值，当这个数据集的MSE或者基尼系数小于这个阈值时，就不再分裂，停止生长。
      
***
        
+ **树的剪枝**
   
     可以想象，如何让树无限的生长下去，最终会对每一条数据都生成一个叶子节点，这就导致了过拟合，降低了树的泛化能力。因此需要进行树的剪枝。所谓剪枝就是将某些已经分裂的数据集合，不让他分裂了。
     
     剪枝策略主要分为两类：从上往下剪枝和从下往上剪枝。上往下剪枝也就是从根开始，遍历所有节点进行剪枝，称为预剪枝；从下往上剪枝是从叶节点开始从下往上剪枝，称为后剪枝。
     
     1. 预剪枝是在决策树生成过程中，对树进行剪枝，提前结束树的分支生长。
     2. 后剪枝是在决策树生长完成之后，对树进行剪枝，得到简化版的决策树。
     
     其实上文的停止生长，也可看作预剪枝，但是这样设置的目的不是剪枝，而是减小树的规模，提高后面真正剪枝的计算效率。这里主要讲述比较常用的后剪枝策略：代价复杂剪枝。所谓代价复杂剪枝就是在树的规模(复杂)和树的误差(代价)之间寻求一个平衡。树规模越大，树的误差越小。树的规模越小，树的误差越大。
     
     我们的目的是为了降低代价复杂度。令其为Ca(T)，其中T代表树。我们用树的叶子节点的个数描述树的规模，记为|T|， 树T的误差记为E(T)：
     
     <a href="http://www.codecogs.com/eqnedit.php?latex={\color{DarkRed}&space;\mathbf{Ca(T)=E(T)&plus;a*|T|}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?{\color{DarkRed}&space;\mathbf{Ca(T)=E(T)&plus;a*|T|}}" title="{\color{DarkRed} \mathbf{Ca(T)=E(T)+a*|T|}}" /></a>
     
     其中 **a** 为权重，**a** 越大，则树规模越小，树的误差越大；**a** 越小，则树规模越大，树的误差越小；下面给出步骤：
     
     1. 遍历树的所有内部节点node(包括根节点)，计算
     
     
                  


