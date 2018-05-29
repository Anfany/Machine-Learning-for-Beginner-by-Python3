# Linear_Regression_Theory

+ **线性回归问题说明**

有<img src="http://latex.codecogs.com/gif.latex?N" title="N" />个样本<img src="http://latex.codecogs.com/gif.latex?(X_{i},&space;Y_{i}),&space;i\in&space;(1,2\cdots&space;N)" title="(X_{i}, Y_{i}), i\in (1,2\cdots N)" />，

<img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},&space;X_{i}^{2},&space;\cdots&space;X_{i}^{m-1}]" title="X_{i} = [X_{i}^{1}, X_{i}^{2}, \cdots X_{i}^{m-1}]" />表示每个样本有<img src="http://latex.codecogs.com/gif.latex?m-1" title="m-1" />个特征。
 
现在为每个样本均添加一个值为1的特征，

也就是令<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" title="X_{i} = [X_{i}^{1},X_{i}^{2},\cdots X_{i}^{m-1},X_{i}^{m}]" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}^{m}&space;=&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}^{m}&space;=&space;1" title="X_{i}^{m} = 1" /></a>。

这样操作本质就是将<a href="http://www.codecogs.com/eqnedit.php?latex=w*X&space;&plus;&space;b" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w*X&space;&plus;&space;b" title="w*X + b" /></a>中的<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?b" title="b" /></a>合为一个<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>，便于计算。

要拟合的回归方程形式为<a href="http://www.codecogs.com/eqnedit.php?latex=\bar{Y}&space;=&space;X&space;\cdot&space;W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bar{Y}&space;=&space;X&space;\cdot&space;W" title="\bar{Y} = X \cdot W" /></a>，其中

<a href="http://www.codecogs.com/eqnedit.php?latex=X&space;=&space;\begin{bmatrix}&space;X_{1}^{1},X_{1}^{2},\cdots&space;,X_{1}^{m}\\&space;X_{2}^{1},X_{2}^{2},\cdots&space;,X_{2}^{m}\\&space;\vdots\\&space;X_{N}^{1},X_{N}^{2},\cdots&space;,X_{N}^{m}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X&space;=&space;\begin{bmatrix}&space;X_{1}^{1},X_{1}^{2},\cdots&space;,X_{1}^{m}\\&space;X_{2}^{1},X_{2}^{2},\cdots&space;,X_{2}^{m}\\&space;\vdots\\&space;X_{N}^{1},X_{N}^{2},\cdots&space;,X_{N}^{m}\\&space;\end{bmatrix}" title="X = \begin{bmatrix} X_{1}^{1},X_{1}^{2},\cdots ,X_{1}^{m}\\ X_{2}^{1},X_{2}^{2},\cdots ,X_{2}^{m}\\ \vdots\\ X_{N}^{1},X_{N}^{2},\cdots ,X_{N}^{m}\\ \end{bmatrix}" /></a>，<a href="http://www.codecogs.com/eqnedit.php?latex=Y&space;=&space;\begin{bmatrix}&space;Y_{1}\\&space;Y_{2}\\&space;\vdots\\&space;Y_{N}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y&space;=&space;\begin{bmatrix}&space;Y_{1}\\&space;Y_{2}\\&space;\vdots\\&space;Y_{N}\\&space;\end{bmatrix}" title="Y = \begin{bmatrix} Y_{1}\\ Y_{2}\\ \vdots\\ Y_{N}\\ \end{bmatrix}" /></a>，<a href="http://www.codecogs.com/eqnedit.php?latex=W&space;=&space;\begin{bmatrix}&space;W_{1}\\&space;W_{2}\\&space;\vdots&space;\\&space;W_{m}\\&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W&space;=&space;\begin{bmatrix}&space;W_{1}\\&space;W_{2}\\&space;\vdots&space;\\&space;W_{m}\\&space;\end{bmatrix}" title="W = \begin{bmatrix} W_{1}\\ W_{2}\\ \vdots \\ W_{m}\\ \end{bmatrix}" /></a>

得到的<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>要使得最小二乘代价函数

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathit{cost}&space;=&space;\frac{1}{2N}\sum_{i=1}^{N}(\bar{Y_{i}}&space;-&space;Y_{i})^{2}=\frac{1}{2N}\sum_{i=1}^{N}(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}=\frac{1}{2N}(X\cdot&space;W-Y)^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathit{cost}&space;=&space;\frac{1}{2N}\sum_{i=1}^{N}(\bar{Y_{i}}&space;-&space;Y_{i})^{2}=\frac{1}{2N}\sum_{i=1}^{N}(X_{i}\cdot&space;W&space;-&space;Y_{i})^{2}=\frac{1}{2N}(X\cdot&space;W-Y)^{2}" title="\mathit{cost} = \frac{1}{2N}\sum_{i=1}^{N}(\bar{Y_{i}} - Y_{i})^{2}=\frac{1}{2N}\sum_{i=1}^{N}(X_{i}\cdot W - Y_{i})^{2}=\frac{1}{2N}(X\cdot W-Y)^{2}" /></a>

最小。


+ **正规方程推导**

由上可知，也就是使<a href="http://www.codecogs.com/eqnedit.php?latex=(X\cdot&space;W&space;-&space;Y)^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(X\cdot&space;W&space;-&space;Y)^{2}" title="(X\cdot W - Y)^{2}" /></a>的值最小。令

<a href="http://www.codecogs.com/eqnedit.php?latex=C&space;=&space;(X\cdot&space;W&space;-&space;Y)^{2}\\&space;........&space;=(X&space;\cdot&space;W&space;-&space;Y)^{T}\cdot&space;(X&space;\cdot&space;W&space;-&space;Y)\\&space;........=W^{T}\cdot&space;X^{T}\cdot&space;X\cdot&space;W-W^{T}\cdot&space;X^{T}\cdot&space;Y-Y^{T}\cdot&space;X\cdot&space;W&space;&plus;&space;Y^{T}\cdot&space;Y" target="_blank"><img src="http://latex.codecogs.com/gif.latex?C&space;=&space;(X\cdot&space;W&space;-&space;Y)^{2}\\&space;........&space;=(X&space;\cdot&space;W&space;-&space;Y)^{T}\cdot&space;(X&space;\cdot&space;W&space;-&space;Y)\\&space;........=W^{T}\cdot&space;X^{T}\cdot&space;X\cdot&space;W-W^{T}\cdot&space;X^{T}\cdot&space;Y-Y^{T}\cdot&space;X\cdot&space;W&space;&plus;&space;Y^{T}\cdot&space;Y" title="C = (X\cdot W - Y)^{2}\\ ........ =(X \cdot W - Y)^{T}\cdot (X \cdot W - Y)\\ ........=W^{T}\cdot X^{T}\cdot X\cdot W-W^{T}\cdot X^{T}\cdot Y-Y^{T}\cdot X\cdot W + Y^{T}\cdot Y" /></a>

计算导数，令其等于0，也就是有

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{d&space;\mathit{C}}{dW}&space;=2&space;\times&space;X^{T}\cdot&space;X\cdot&space;W&space;-X^{T}\cdot&space;Y-(Y^{T}\cdot&space;X)^{T}\\&space;............=2&space;\times(&space;X^{T}\cdot&space;X\cdot&space;W&space;-X^{T}\cdot&space;Y)\\&space;............=0\\&space;\\&space;........\Rightarrow&space;X^{T}\cdot&space;X\cdot&space;W&space;=X^{T}\cdot&space;Y\\&space;........\Rightarrow&space;\mathbf{W&space;=&space;(X^{T}\cdot&space;X)^{-1}\cdot&space;X^{T}\cdot&space;Y&space;}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{d&space;\mathit{C}}{dW}&space;=2&space;\times&space;X^{T}\cdot&space;X\cdot&space;W&space;-X^{T}\cdot&space;Y-(Y^{T}\cdot&space;X)^{T}\\&space;............=2&space;\times(&space;X^{T}\cdot&space;X\cdot&space;W&space;-X^{T}\cdot&space;Y)\\&space;............=0\\&space;\\&space;........\Rightarrow&space;X^{T}\cdot&space;X\cdot&space;W&space;=X^{T}\cdot&space;Y\\&space;........\Rightarrow&space;\mathbf{W&space;=&space;(X^{T}\cdot&space;X)^{-1}\cdot&space;X^{T}\cdot&space;Y&space;}" title="\frac{d \mathit{C}}{dW} =2 \times X^{T}\cdot X\cdot W -X^{T}\cdot Y-(Y^{T}\cdot X)^{T}\\ ............=2 \times( X^{T}\cdot X\cdot W -X^{T}\cdot Y)\\ ............=0\\ \\ ........\Rightarrow X^{T}\cdot X\cdot W =X^{T}\cdot Y\\ ........\Rightarrow \mathbf{W = (X^{T}\cdot X)^{-1}\cdot X^{T}\cdot Y }" /></a>


+ **梯度下降推导**

计算代价函数<a href="http://www.codecogs.com/eqnedit.php?latex=\mathit{cost}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathit{cost}" title="\mathit{cost}" /></a>对<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>的梯度：

<a href="http://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;\mathit{cost}&space;=&space;\frac{1}{2N}&space;\bigtriangledown&space;\mathit{C}\\&space;...............=\frac{1}{N}(X^{T}\cdot&space;X\cdot&space;W-X^{T}\cdot&space;Y)\\&space;...............=\frac{1}{N}(X^{T}\cdot&space;(X\cdot&space;W&space;-Y))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\bigtriangledown&space;\mathit{cost}&space;=&space;\frac{1}{2N}&space;\bigtriangledown&space;\mathit{C}\\&space;...............=\frac{1}{N}(X^{T}\cdot&space;X\cdot&space;W-X^{T}\cdot&space;Y)\\&space;...............=\frac{1}{N}(X^{T}\cdot&space;(X\cdot&space;W&space;-Y))" title="\bigtriangledown \mathit{cost} = \frac{1}{2N} \bigtriangledown \mathit{C}\\ ...............=\frac{1}{N}(X^{T}\cdot X\cdot W-X^{T}\cdot Y)\\ ...............=\frac{1}{N}(X^{T}\cdot (X\cdot W -Y))" /></a>

更新<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>：

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{W&space;=&space;W&space;-&space;\boldsymbol{\eta}&space;\times&space;\boldsymbol{\bigtriangledown&space;\mathit{cost}}&space;}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{W&space;=&space;W&space;-&space;\boldsymbol{\eta}&space;\times&space;\boldsymbol{\bigtriangledown&space;\mathit{cost}}&space;}" title="\mathbf{W = W - \boldsymbol{\eta} \times \boldsymbol{\bigtriangledown \mathit{cost}} }" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=\eta" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\eta" title="\eta" /></a>表示学习率。
