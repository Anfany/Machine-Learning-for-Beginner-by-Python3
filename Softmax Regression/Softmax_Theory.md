# Softmax Theory


+ **<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{Softmax}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{Softmax}" title="\mathbf{Softmax}" /></a>回归问题说明**

有<img src="http://latex.codecogs.com/gif.latex?N" title="N" />个样本，<img src="http://latex.codecogs.com/gif.latex?(X_{i},&space;Y_{i}),&space;i\in&space;(1,2\cdots&space;N)" title="(X_{i}, Y_{i}), i\in (1,2\cdots N)" />，<img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},&space;X_{i}^{2},&space;\cdots&space;X_{i}^{m-1}]" title="X_{i} = [X_{i}^{1}, X_{i}^{2}, \cdots X_{i}^{m-1}]" />， 表示每个样本有<img src="http://latex.codecogs.com/gif.latex?m-1" title="m-1" />个特征；<a href="http://www.codecogs.com/eqnedit.php?latex=Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" title="Y_{i}\in \left \{ 1,2,\cdots ,K \right \}" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=K\geq&space;2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K\geq&space;2" title="K\geq 2" /></a>，代表类别数。
 
现在为每个样本添加一个特征值为1的特征，也就是令<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" title="X_{i} = [X_{i}^{1},X_{i}^{2},\cdots X_{i}^{m-1},X_{i}^{m}]" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}^{m}&space;=&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}^{m}&space;=&space;1" title="X_{i}^{m} = 1" /></a>。这样操作本质就是将<a href="http://www.codecogs.com/eqnedit.php?latex=w*X&space;&plus;&space;b" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w*X&space;&plus;&space;b" title="w*X + b" /></a>中的<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?b" title="b" /></a>合为一个<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>，便于计算。

和逻辑回归类似，同样引入<a href="http://www.codecogs.com/eqnedit.php?latex=Sigmoid" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Sigmoid" title="Sigmoid" /></a>函数：

<a href="http://www.codecogs.com/eqnedit.php?latex=\Phi(x)=\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Phi(x)=\frac{1}{1&plus;e^{-x}}" title="\Phi(x)=\frac{1}{1+e^{-x}}" /></a>


<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{Softmax}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{Softmax}" title="\mathbf{Softmax}" /></a>回归中，样本<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}" title="X_{i}" /></a>属于<a href="http://www.codecogs.com/eqnedit.php?latex=Y_{g}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y_{g}" title="Y_{g}" /></a>类别的概率表达式为：

<a href="http://www.codecogs.com/eqnedit.php?latex=P(Y=Y_{g}|X_{i},W)=\frac{\Phi&space;(X_{i}\cdot&space;W_{g})}{\sum_{j=1}^{K}\Theta&space;(X_{i}\cdot&space;W_{j})}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(Y=Y_{g}|X_{i},W)=\frac{\Phi&space;(X_{i}\cdot&space;W_{g})}{\sum_{j=1}^{K}\Theta&space;(X_{i}\cdot&space;W_{j})}" title="P(Y=Y_{g}|X_{i},W)=\frac{\Phi (X_{i}\cdot W_{g})}{\sum_{j=1}^{K}\Theta (X_{i}\cdot W_{j})}" /></a>





