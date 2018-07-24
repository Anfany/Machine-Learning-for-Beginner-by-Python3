# SVM Theory

* **分类**
    * **引入**
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/svm.png)
    
    如上图左，平面内展示了二维数据样本，正例用“+”号表示，负例用“-”号表示。存在很多的分割线可以分开上述两类(上图右)。但只存在一条最优的：**最近负例到这条线的距离等于到最近正例到这条线的距离，并且两个距离的和是所有线中最大的**(见下图左)。
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/zuiyou.png)
    
    下面定义一个向量**W**，现在只定义它的方向，就是与分割线垂直。假设这个样本空间中存在一个样本向量**U**，**PP=U·W**可看作向量**U**在**W**上的投影(因为**W**未定义长度)。现在规定：如果**U**是一个负例，则PP的值越小越好；如果是一个正例，则PP的值越大越好(见上图右)。
   
    因此在判断样本属于正例还是负例时，就是判断PP的值是否大于一个数，也就是判断**U·W**+**b>=0**。为了求得**W**和**b**，需要添加一些约束条件。不失一般性，对于一个正例**Xz**，可以令**Xz·W**+**b>=1**， 对于一个负例**Xf**，可以令**Xf·W**+**b<=-1**。数学便利性：对于正例，令**Y= 1**，对于负例，令**Y=-1**。因此结合以上式子可得：对于任意的样本h，有**Yh(Xh·W**+**b) >=1**，也就是**Yh(Xh·W**+**b)-1 >=0**。
    
    现在着重研究下**Yh(Xh·W**+**b)-1 = 0**的情况，当样本h在上边缘或者在下边缘时，式子成立。假设正例**Xz**，负例**Xf**分别在上边缘、下边缘上。则上边缘与下边缘的距离，也就是分割线的**最大间隔**为<img src="http://latex.codecogs.com/svg.latex?g=(Xz-Xf)\cdot\frac{W}{||W||}" border="0"/>
    
    因为**Yz=1，Xz·W = 1-b， Yf=-1，Xf·W = -1-b**， 所以**g=2/||W||**。
    
  ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/gap.png)
  
  OK，整理下思路。我们需要求得**g**最大，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{2}{||W||}" border="0"/>最大，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{1}{||W||}" border="0"/>最大，也就是<img src="http://latex.codecogs.com/svg.latex?||W||" border="0"/>最小，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{1}{2}||W||^{2}" border="3"/>最小。
  
   * **线性可分情况：硬间隔**
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/pro1.png)
  
  这是一个凸二次优化问题，可以求得其全局最小值。引入拉格朗日乘子，可得到拉格朗日函数：
  
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/mubiao.png)
  
   也就是计算上式的极小值，求极值，就需要求导，并且另导数等于0，得到下面的式子：
 
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/der.png)
   
  将上面的结果带入到拉格朗日函数中，得到：
  
  ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/computer.png)
  
  也就将原始问题转为其对偶问题：
  
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/duiou1.png)
  
 
   * **线性不可分情况：软间隔**
   
   上面描述的是线性可分的情形，也就是存在线或者面可以将样本按类别很好的分开。现在看下面存在离群点的2种情况：
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/ying.png)
   
   情况A：如果依然按着硬间隔进行划分，可能会过拟合；情况B：不存在硬间隔。因此在这种情形下，要适当的对约束条件进行如下的放宽。
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/ru.png)
   
   对于硬间隔而言，**Yh(Xh·W**+**b) >=0**对任何的样本均成立，也就是保证所有的样本均分类正确。而在软间隔中，**Yh(Xh·W**+**b) >=0**不是对所有的样本均成立，也就是允许对一些离散的点分类错误。
   
   因为不能无限放宽，因此需要在目标函数里面增加一个惩罚项，问题变为：
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/ruan.png)
   
   上式中C的值越大，离群点的存在会使得目标函数的值变大，为了降低目标函数的值，这时候，得到的结果趋向于硬间隔。
   
   经过和硬间隔同样的变化，得到软间隔的对偶问题：
   
   ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/duiou2.png)
   
   模型多了αi<=C的限制条件，并且表达式中没有参数ξi，此时b的求值公式也会发生相应的改变。
   
     * **线性不可分情况：核函数**
     
     以上描述的都是线性可分的情形，现在考虑下面的情形(下图左)：
     
     ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/noli.png)
     
     首先上述不存在硬间隔，如果使用软间隔，则和数据分布不符，因此需要对原始的数据进行非线性转换(上图右)。
     
     前面2种情况的最终的对偶问题表达式中，都只是和内积有关。因此只要找到一个函数Ｆ，使得 
     
     ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/neiji.png)
     
     其中P是非线性转换函数。也就是说F的值恰好是非线性转换后的向量的内积，此时的F称为核函数。
   
     下面介绍几种常用的核函数：
   
     ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/he.png)
     
     经过核函数的对偶问题为：
     
     ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/duiou3.png)
    
   
   * **求解算法：序列最小优化算法SMO**
   
       + **坐标上升/下降算法**
       
       坐标上升用于求极大值，下降用于计算极小值，两者的本质是一样的。对于凸函数而言，这种方法可以获得全局最优值，但对于有多个极值的函数而言，很大可能会获得局部最优值，这和初始值的选取有很大关系。下面用例子说明算法的步骤：
       
        ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/formula/zuo.png)
      重复步骤1和2，直到函数的值变化很小。[基于Python3实现的上述例子的代码](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/coor_de.py)
     
     **图示**：
     
      ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/biao.png)

     + **SMO算法**





    
       
       
       
       
      
   
   
   
   
   
   
   
   
   
   
   
   
