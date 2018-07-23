# SVM Theory

* **分类**
    * **引入**
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/svm.png)
    
    如上图左，平面内展示了二维数据样本，正例用“+”号表示，负例用“-”号表示。存在很多的分割线可以分开上述两类(上图右)。但只存在一条最优的：**最近负例到这条线的距离等于到最近正例到这条线的距离，并且两个距离的和是所有线中最大的**(见下图左)。
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/zuiyou.png)
    
    下面定义一个向量**W**，现在只定义它的方向，就是与分割线垂直。假设这个样本空间中存在一个样本向量**U**，**PP=U·W**可看作向量**U**在**W**上的投影(因为**W**未定义长度)。现在规定：如果**U**是一个负例，则PP的值越小越好；如果是一个正例，则PP的值越大越好(见上图右)。
   
    因此在判断样本属于正例还是负例时，就是判断PP的值是否大于一个数，也就是判断**U·W**+**b>=0**。为了求得**W**和**b**，需要添加一些约束条件。不失一般性，对于一个正例**Xz**，可以令**Xz·W**+**b>=1**， 对于一个负例**Xf**，可以令**Xf·W**+**b<=-1**。数学便利性：对于正例，令**Y= 1**，对于负例，令**Y=-1**。因此结合以上式子可得：对于任意的样本h，有**Yh(Xh·W**+**b)-1 >=0**。
    
    现在着重研究下**Yh(Xh·W**+**b)-1 = 0**的情况，当样本h在上边缘或者在下边缘时，式子成立。假设正例**Xz**，负例**Xf**分别在上边缘、下边缘上。则上边缘与下边缘的距离，也就是分割线的**最大间隔**为<img src="http://latex.codecogs.com/svg.latex?g=(Xz-Xf)\cdot\frac{W}{||W||}" border="0"/>
    
    因为**Yz=1，Xz·W = 1-b， Yf=-1，Xf·W = -1-b**， 所以**g=2/||W||**。
    
  ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/gap.png)
  
  OK，整理下思路。我们需要求得**g**最大，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{2}{||W||}" border="0"/>最大，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{1}{||W||}" border="0"/>最大，也就是<img src="http://latex.codecogs.com/svg.latex?\||W||" border="0"/>最小，也就是<img src="http://latex.codecogs.com/svg.latex?\frac{1}{2}W^{2}" border="0"/>最小。
