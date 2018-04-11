# Kmeans++ Theory
-------------


+ **Kmeans 聚类**

样本集合<a href="http://www.codecogs.com/eqnedit.php?latex=X=\begin{pmatrix}&space;X_{1},&space;&X_{2},&space;&&space;\cdots&space;,&&space;X_{N}&space;\end{pmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X=\begin{pmatrix}&space;X_{1},&space;&X_{2},&space;&&space;\cdots&space;,&&space;X_{N}&space;\end{pmatrix}" title="X=\begin{pmatrix} X_{1}, &X_{2}, & \cdots ,& X_{N} \end{pmatrix}" /></a>， <a href="http://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="http://latex.codecogs.com/gif.latex?N" title="N" /></a>为样本个数。<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}=\begin{bmatrix}&space;X_{i}^{1}&space;&&space;X_{i}^{2}&\cdots&space;&&space;X_{i}^{m}&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}=\begin{bmatrix}&space;X_{i}^{1}&space;&&space;X_{i}^{2}&\cdots&space;&&space;X_{i}^{m}&space;\end{bmatrix}" title="X_{i}=\begin{bmatrix} X_{i}^{1} & X_{i}^{2}&\cdots & X_{i}^{m} \end{bmatrix}" /></a>， <a href="http://www.codecogs.com/eqnedit.php?latex=m" target="_blank"><img src="http://latex.codecogs.com/gif.latex?m" title="m" /></a>为每个样本的特征个数。

要把这些样本分为<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>类，<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>个类别的样本集合为<a href="http://www.codecogs.com/eqnedit.php?latex=C=\begin{bmatrix}&space;C1&space;&&space;C2&&space;\cdots&space;&&space;Ck&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?C=\begin{bmatrix}&space;C1&space;&&space;C2&&space;\cdots&space;&&space;Ck&space;\end{bmatrix}" title="C=\begin{bmatrix} C1 & C2& \cdots & Ck \end{bmatrix}" /></a>。


得到的<a href="http://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K" title="K" /></a>个类别要使得下式取得最小值：

<a href="http://www.codecogs.com/eqnedit.php?latex=D&space;=&space;\sum_{k=1}^{K}\sum_{i=1}^{nk}dis(X_{ki}-Ckc)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D&space;=&space;\sum_{k=1}^{K}\sum_{i=1}^{nk}dis(X_{ki}-Ckc)" title="D = \sum_{k=1}^{K}\sum_{i=1}^{nk}dis(X_{ki}-Ckc)" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=nk" target="_blank"><img src="http://latex.codecogs.com/gif.latex?nk" title="nk" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的样本集合中样本的个数；<a href="http://www.codecogs.com/eqnedit.php?latex=Ckc" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Ckc" title="Ckc" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的中心；<a href="http://www.codecogs.com/eqnedit.php?latex=X_{ki}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{ki}" title="X_{ki}" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的样本集合中的第<a href="http://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?i" title="i" /></a>个样本。<a href="http://www.codecogs.com/eqnedit.php?latex=dis(X_{ki}-Ckc)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?dis(X_{ki}-Ckc)" title="dis(X_{ki}-Ckc)" /></a>表示第<a href="http://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="http://latex.codecogs.com/gif.latex?k" title="k" /></a>个类别的样本集合中的样本与该类别中心的距离。

 
+ **不同的距离计算方式**

有两个点<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{A,B}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{A,B}" title="\mathbf{A,B}" /></a>, 坐标分别为<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{A}=[a1,a2,\cdots,am]\begin{matrix}&space;&&space;\end{matrix}&space;\mathbf{B}=[b1,b2,\cdots,bm]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{A}=[a1,a2,\cdots,am]\begin{matrix}&space;&&space;\end{matrix}&space;\mathbf{B}=[b1,b2,\cdots,bm]" title="\mathbf{A}=[a1,a2,\cdots,am]\begin{matrix} & \end{matrix} \mathbf{B}=[b1,b2,\cdots,bm]" /></a>
        
   + 欧几里得距离
   
   <a href="http://www.codecogs.com/eqnedit.php?latex=E(\mathbf{A,B})&space;=&space;\sqrt{\sum_{i=1}^{m}(ai-bi)^{2}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?E(\mathbf{A,B})&space;=&space;\sqrt{\sum_{i=1}^{m}(ai-bi)^{2}}" title="E(\mathbf{A,B}) = \sqrt{\sum_{i=1}^{m}(ai-bi)^{2}}" /></a>
   
   
   + 曼哈顿距离
   
   <a href="http://www.codecogs.com/eqnedit.php?latex=M(\mathbf{A,B})&space;=&space;\sum_{i=1}^{m}\left&space;|&space;ai-bi&space;\right&space;|" target="_blank"><img src="http://latex.codecogs.com/gif.latex?M(\mathbf{A,B})&space;=&space;\sum_{i=1}^{m}\left&space;|&space;ai-bi&space;\right&space;|" title="M(\mathbf{A,B}) = \sum_{i=1}^{m}\left | ai-bi \right |" /></a>
   
   
   + 闵可夫斯基距离
  
   <a href="http://www.codecogs.com/eqnedit.php?latex=M(\mathbf{A,B})&space;=&space;\begin{pmatrix}&space;\sum_{i=1}^{m}(ai-bi)^{p}&space;\end{pmatrix}&space;^{1/p}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?M(\mathbf{A,B})&space;=&space;\begin{pmatrix}&space;\sum_{i=1}^{m}(ai-bi)^{p}&space;\end{pmatrix}&space;^{1/p}" title="M(\mathbf{A,B}) = \begin{pmatrix} \sum_{i=1}^{m}(ai-bi)^{p} \end{pmatrix} ^{1/p}" /></a>
    
    
    
            
            
        

            
            


