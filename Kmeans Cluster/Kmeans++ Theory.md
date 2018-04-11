# Kmeans++ Theory
-------------


+ Kmeans 聚类

样本集合<a href="http://www.codecogs.com/eqnedit.php?latex=X=\begin{bmatrix}&space;X_{1}\\&space;X_{2}\\&space;\vdots&space;\\&space;X_{N}&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X=\begin{bmatrix}&space;X_{1}\\&space;X_{2}\\&space;\vdots&space;\\&space;X_{N}&space;\end{bmatrix}" title="X=\begin{bmatrix} X_{1}\\ X_{2}\\ \vdots \\ X_{N} \end{bmatrix}" /></a>， <a href="http://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="http://latex.codecogs.com/gif.latex?N" title="N" /></a>为样本个数。<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}=\begin{bmatrix}&space;X_{i}^{1}&space;&&space;X_{i}^{2}&\cdots&space;&&space;X_{i}^{m}&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}=\begin{bmatrix}&space;X_{i}^{1}&space;&&space;X_{i}^{2}&\cdots&space;&&space;X_{i}^{m}&space;\end{bmatrix}" title="X_{i}=\begin{bmatrix} X_{i}^{1} & X_{i}^{2}&\cdots & X_{i}^{m} \end{bmatrix}" /></a>， <a href="http://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="http://latex.codecogs.com/gif.latex?m" title="m" /></a>为每个样本的特征个数。

要把这些样本分为<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>类，<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>个类别的样本集合为<a href="http://www.codecogs.com/eqnedit.php?latex=C=\begin{bmatrix}&space;C1&space;&&space;C2&&space;\cdots&space;&&space;Ck&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?C=\begin{bmatrix}&space;C1&space;&&space;C2&&space;\cdots&space;&&space;Ck&space;\end{bmatrix}" title="C=\begin{bmatrix} C1 & C2& \cdots & Ck \end{bmatrix}" /></a>。


得到的<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>个类别要使得下式取得最小值：

<a href="http://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\sum_{k=1}^{K}\sum_{i=1}^{nk}dis(x_{i}-Ckc)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D&space;=&space;\sum_{k=1}^{K}\sum_{i=1}^{nk}dis(x_{i}-Ckc)" title="D = \sum_{k=1}^{K}\sum_{i=1}^{nk}dis(X_{i}-Ckc)" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=nk" target="_blank"><img src="http://latex.codecogs.com/gif.latex?nk" title="nk" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的样本集合中样本的个数；<a href="http://www.codecogs.com/eqnedit.php?latex=Ckc" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Ckc" title="Ckc" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的中心；<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}" title="X_{i}" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的样本集合中的样本。<a href="http://www.codecogs.com/eqnedit.php?latex=dis(X_{i}-Ckc)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?dis(X_{i}-Ckc)" title="dis(X_{i}-Ckc)" /></a>表示某个类别的样本集合中的样本与该类别中心的距离。



+ Kmeans++ 聚类
