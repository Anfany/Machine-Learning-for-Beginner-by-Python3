# SVM Theory

* **分类**
    * **引入**
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/svm.png)
    
    如上图左，平面内展示了二维数据样本，正例用“+”号表示，负例用“-”号表示。存在很多的分割线可以分开上述两类(上图右)。但只存在一条最优的：**最近负例到这条线的距离等于到最近正例到这条线的距离，并且两个距离的和是所有线中最大的**(见下图左)。
    
    ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/SVM/zuijia.png)
    
    下面定义一个向量**W**，现在只定义它的方向，就是与分割线垂直。假设这个样本空间中存在一个样本向量**U**，**point=U·W**可看作向量**U**在**W**上的投影(因为**W**未定义长度)。现在规定：如果**U**是一个负例，则point的值越小越好；如果是一个正例，则point的值越大越好(见上图右)。
   
    不失一般性，对于一个正例**Xz**，可以令**Xz·W**+**b>=1**， 对于一个负例**Xf**，可以令**Xf·W**+**b<=-1**。数学便利性：对于正例**Y= 1**，对于负例**Y=-1**。因此结合以上式子可得：
    对于样本h，有**Yh(Xh·W**+**b)>=1**
    
