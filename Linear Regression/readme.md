 # Linear Regression
 

 #### 正规方程推导
 
 有<img src="http://latex.codecogs.com/gif.latex?N" title="N" />个样本，<img src="http://latex.codecogs.com/gif.latex?(X_{i},&space;Y_{i}),&space;i\in&space;(1,2\cdots&space;N)" title="(X_{i}, Y_{i}), i\in (1,2\cdots N)" />，<img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},&space;X_{i}^{2},&space;\cdots&space;X_{i}^{m-1}]" title="X_{i} = [X_{i}^{1}, X_{i}^{2}, \cdots X_{i}^{m-1}]" />， 表示每个样本有<img src="http://latex.codecogs.com/gif.latex?m-1" title="m-1" />个特征。
 
现在为每个样本添加一个特征值为1的特征，也就是令<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" title="X_{i} = [X_{i}^{1},X_{i}^{2},\cdots X_{i}^{m-1},X_{i}^{m}]" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}^{m}&space;=&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}^{m}&space;=&space;1" title="X_{i}^{m} = 1" /></a>。这样操作本质就是将<a href="http://www.codecogs.com/eqnedit.php?latex=w*X&space;&plus;&space;b" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w*X&space;&plus;&space;b" title="w*X + b" /></a>中的<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?b" title="b" /></a>合为一个<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>，便于计算。

要拟合的回归方程形式为<a href="http://www.codecogs.com/eqnedit.php?latex=\bar{Y}&space;=&space;X&space;\cdot&space;W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bar{Y}&space;=&space;X&space;\cdot&space;W" title="\bar{Y} = X \cdot W" /></a>，其中

<a href="http://www.codecogs.com/eqnedit.php?latex=X&space;=&space;\begin{bmatrix}&space;X_{1}^{1},X_{1}^{2},\cdots&space;,X_{1}^{m}\\&space;X_{2}^{1},X_{2}^{2},\cdots&space;,X_{2}^{m}\\&space;\vdots\\&space;X_{N}^{1},X_{N}^{2},\cdots&space;,X_{N}^{m}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X&space;=&space;\begin{bmatrix}&space;X_{1}^{1},X_{1}^{2},\cdots&space;,X_{1}^{m}\\&space;X_{2}^{1},X_{2}^{2},\cdots&space;,X_{2}^{m}\\&space;\vdots\\&space;X_{N}^{1},X_{N}^{2},\cdots&space;,X_{N}^{m}\\&space;\end{bmatrix}" title="X = \begin{bmatrix} X_{1}^{1},X_{1}^{2},\cdots ,X_{1}^{m}\\ X_{2}^{1},X_{2}^{2},\cdots ,X_{2}^{m}\\ \vdots\\ X_{N}^{1},X_{N}^{2},\cdots ,X_{N}^{m}\\ \end{bmatrix}" /></a>，<a href="http://www.codecogs.com/eqnedit.php?latex=Y&space;=&space;\begin{bmatrix}&space;Y_{1}\\&space;Y_{2}\\&space;\vdots\\&space;Y_{N}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y&space;=&space;\begin{bmatrix}&space;Y_{1}\\&space;Y_{2}\\&space;\vdots\\&space;Y_{N}\\&space;\end{bmatrix}" title="Y = \begin{bmatrix} Y_{1}\\ Y_{2}\\ \vdots\\ Y_{N}\\ \end{bmatrix}" /></a>，<a href="http://www.codecogs.com/eqnedit.php?latex=W&space;=&space;\begin{bmatrix}&space;W_{1}\\&space;W_{2}\\&space;\vdots&space;\\&space;W_{m}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W&space;=&space;\begin{bmatrix}&space;W_{1}\\&space;W_{2}\\&space;\vdots&space;\\&space;W_{m}\\&space;\end{bmatrix}" title="W = \begin{bmatrix} W_{1}\\ W_{2}\\ \vdots \\ W_{m}\\ \end{bmatrix}" /></a>

得到的<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>要使得

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathit{cost}&space;=\frac{1}{2N}&space;\sum_{i&space;=&space;1}^{N}(\bar{Y}_{i}&space;-&space;Y_{i})^{2}=\frac{1}{2N}&space;\sum_{i&space;=&space;1}^{N}(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathit{cost}&space;=\frac{1}{2N}&space;\sum_{i&space;=&space;1}^{N}(\bar{Y}_{i}&space;-&space;Y_{i})^{2}=\frac{1}{2N}&space;\sum_{i&space;=&space;1}^{N}(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}" title="\mathit{cost} =\frac{1}{2N} \sum_{i = 1}^{N}(\bar{Y}_{i} - Y_{i})^{2}=\frac{1}{2N} \sum_{i = 1}^{N}(X_{i}\cdot W - Y_{i})^{2}" /></a>

最小，也就是使得<a href="http://www.codecogs.com/eqnedit.php?latex=(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}" title="(X_{i}\cdot W - Y_{i})^{2}" /></a>最小。



 #### 文件说明
 
 + 数据文件
 
     + 波士顿房价数据集：[Boston.csv](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Boston.csv)
     
     + 数据集[说明](http://lib.stat.cmu.edu/datasets/boston)
 
+ 基于不同库的代码文件
 
     + Sklearn：[Linear_Regression_Sklearn.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Linear_Regression_Sklearn.py)
 
     + TensorFlow：[Linear_Regression_TensorFlow.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Linear_Regression_TensorFlow.py)
 
     + AnFany：[Linear_Regression_AnFany.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Linear_Regression_AnFany.py)
 
 + 辅助代码文件
 
     + 波士顿房价数据集爬虫程序：[Boston_Spyder.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Boston_Spyder.py)
 
     + 数据读取与预处理程序：[Boston_Data.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Boston_Data.py)
 
 
 + 结果展示
 
     + 对比程序中用到的TensorFlow程序：[TensorFlow_rewrite.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/TensorFlow_rewrite.py)
 
     + 一元回归对比程序：[Linear_Regression_Compare.py](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Linear_Regression_Compare.py)
 
     + 一元回归对比图示：
     
     ![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/Linear%20Regression/Linear_Regression.png)
 
 
 
 
 
 
 
 
 
 
 
