# Softmax Theory


+ **Softmax回归问题说明**

有<img src="http://latex.codecogs.com/gif.latex?N" title="N" />个样本，<img src="http://latex.codecogs.com/gif.latex?(X_{i},&space;Y_{i}),&space;i\in&space;(1,2\cdots&space;N)" title="(X_{i}, Y_{i}), i\in (1,2\cdots N)" />，<img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},&space;X_{i}^{2},&space;\cdots&space;X_{i}^{m-1}]" title="X_{i} = [X_{i}^{1}, X_{i}^{2}, \cdots X_{i}^{m-1}]" />， 表示每个样本有<img src="http://latex.codecogs.com/gif.latex?m-1" title="m-1" />个特征；<a href="http://www.codecogs.com/eqnedit.php?latex=Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" title="Y_{i}\in \left \{ 1,2,\cdots ,K \right \}" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=K\geq&space;2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K\geq&space;2" title="K\geq 2" /></a>，代表类别数。
 
现在为每个样本添加一个特征值为1的特征，也就是令<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" title="X_{i} = [X_{i}^{1},X_{i}^{2},\cdots X_{i}^{m-1},X_{i}^{m}]" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}^{m}&space;=&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}^{m}&space;=&space;1" title="X_{i}^{m} = 1" /></a>。这样操作本质就是将<a href="http://www.codecogs.com/eqnedit.php?latex=w*X&space;&plus;&space;b" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w*X&space;&plus;&space;b" title="w*X + b" /></a>中的<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?b" title="b" /></a>合为一个<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>，便于计算。


不同于逻辑回归中的二分类的情况，样本属于某一类别的概率表达式为：




Softmax回归中，样本属于某一类别的概率表达式为：



