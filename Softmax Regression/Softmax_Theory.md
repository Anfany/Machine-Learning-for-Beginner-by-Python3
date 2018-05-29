# Softmax Theory


+ **<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{Softmax}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{Softmax}" title="\mathbf{Softmax}" /></a>回归问题说明**

有<img src="http://latex.codecogs.com/gif.latex?N" title="N" />个样本，<img src="http://latex.codecogs.com/gif.latex?(X_{i},&space;Y_{i}),&space;i\in&space;(1,2\cdots&space;N)" title="(X_{i}, Y_{i}), i\in (1,2\cdots N)" />，<img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},&space;X_{i}^{2},&space;\cdots&space;X_{i}^{m-1}]" title="X_{i} = [X_{i}^{1}, X_{i}^{2}, \cdots X_{i}^{m-1}]" />， 表示每个样本有<img src="http://latex.codecogs.com/gif.latex?m-1" title="m-1" />个特征；<a href="http://www.codecogs.com/eqnedit.php?latex=Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y_{i}\in&space;\left&space;\{&space;1,2,\cdots&space;,K&space;\right&space;\}" title="Y_{i}\in \left \{ 1,2,\cdots ,K \right \}" /></a>，

其中<a href="http://www.codecogs.com/eqnedit.php?latex=K\geq&space;2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?K\geq&space;2" title="K\geq 2" /></a>，代表类别数。
 
现在为每个样本添加一个特征值为1的特征，也就是令<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}&space;=&space;[X_{i}^{1},X_{i}^{2},\cdots&space;X_{i}^{m-1},X_{i}^{m}]" title="X_{i} = [X_{i}^{1},X_{i}^{2},\cdots X_{i}^{m-1},X_{i}^{m}]" /></a>，其中<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}^{m}&space;=&space;1" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}^{m}&space;=&space;1" title="X_{i}^{m} = 1" /></a>。这样操作本质就是将<a href="http://www.codecogs.com/eqnedit.php?latex=w*X&space;&plus;&space;b" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w*X&space;&plus;&space;b" title="w*X + b" /></a>中的<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?w" title="w" /></a>和<a href="http://www.codecogs.com/eqnedit.php?latex=w" target="_blank"><img src="http://latex.codecogs.com/gif.latex?b" title="b" /></a>合为一个<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>，便于计算。

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{Softmax}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{Softmax}" title="\mathbf{Softmax}" /></a>回归中，样本<a href="http://www.codecogs.com/eqnedit.php?latex=X_{i}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X_{i}" title="X_{i}" /></a>属于<a href="http://www.codecogs.com/eqnedit.php?latex=Y_{g}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Y_{g}" title="Y_{g}" /></a>类别的概率表达式为：

<a href="http://www.codecogs.com/eqnedit.php?latex=P(Y=Y_{g}|X_{i},W)=\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{j=1}^{K}e^{X_{i}\cdot&space;W_{j}}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?P(Y=Y_{g}|X_{i},W)=\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{j=1}^{K}e^{X_{i}\cdot&space;W_{j}}}" title="P(Y=Y_{g}|X_{i},W)=\frac{e^{X_{i}\cdot W_{g}}}{\sum_{j=1}^{K}e^{X_{i}\cdot W_{j}}}" /></a>

根据极大似然估计，也就是找到下面式子的最大值：

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{ML(W)}=\prod_{i=1}^{N}\sum_{g=1}^{K}&space;\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{&space;X_{i}\cdot&space;W_{k}}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{ML(W)}=\prod_{i=1}^{N}\sum_{g=1}^{K}&space;\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{&space;X_{i}\cdot&space;W_{k}}}" title="\mathbf{ML(W)}=\prod_{i=1}^{N}\sum_{g=1}^{K} \boldsymbol{I}(Y=Y_{g}|X_{i}) \frac{e^{X_{i}\cdot W_{g}}}{\sum_{k=1}^{K}e^{ X_{i}\cdot W_{k}}}" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=\boldsymbol{I}(Y=Y_{g}|X_{i})=\left\{\begin{matrix}&space;1&space;&&space;\mathrm{if}&space;(X_{i},Y_{g})\in&space;S\\&space;0&space;&&space;\mathrm{if}&space;(X_{i},Y_{g})\notin&space;S&space;\end{matrix}\right." target="_blank"><img src="http://latex.codecogs.com/gif.latex?\boldsymbol{I}(Y=Y_{g}|X_{i})=\left\{\begin{matrix}&space;1&space;&&space;\mathrm{if}&space;(X_{i},Y_{g})\in&space;S\\&space;0&space;&&space;\mathrm{if}&space;(X_{i},Y_{g})\notin&space;S&space;\end{matrix}\right." title="\boldsymbol{I}(Y=Y_{g}|X_{i})=\left\{\begin{matrix} 1 & \mathrm{if} (X_{i},Y_{g})\in S\\ 0 & \mathrm{if} (X_{i},Y_{g})\notin S \end{matrix}\right." /></a>，<a href="http://www.codecogs.com/eqnedit.php?latex=S" target="_blank"><img src="http://latex.codecogs.com/gif.latex?S" title="S" /></a>表示总的样本集合。

上述式子和取对数后的式子增减性相同，下面取其对数：


<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{L(W)}=\mathit{ln}(\mathbf{ML(W)})=\sum_{i=1}^{N}\sum_{g=1}^{K}&space;\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\mathit{ln}\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{&space;X_{i}\cdot&space;W_{k}}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{L(W)}=\mathit{ln}(\mathbf{ML(W)})=\sum_{i=1}^{N}\sum_{g=1}^{K}&space;\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\mathit{ln}\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{&space;X_{i}\cdot&space;W_{k}}}" title="\mathbf{L(W)}=\mathit{ln}(\mathbf{ML(W)})=\sum_{i=1}^{N}\sum_{g=1}^{K} \boldsymbol{I}(Y=Y_{g}|X_{i}) \mathit{ln}\frac{e^{X_{i}\cdot W_{g}}}{\sum_{k=1}^{K}e^{ X_{i}\cdot W_{k}}}" /></a>

构建如下的成本函数<a href="http://www.codecogs.com/eqnedit.php?latex=cost" target="_blank"><img src="http://latex.codecogs.com/gif.latex?cost" title="cost" /></a>，令：

<a href="http://www.codecogs.com/eqnedit.php?latex=cost&space;=&space;-\frac{1}{N}\mathbf{L}(W)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?cost&space;=&space;-\frac{1}{N}\mathbf{L}(W)" title="cost = -\frac{1}{N}\mathbf{L}(W)" /></a>


+ **梯度下降**

则成本函数<a href="http://www.codecogs.com/eqnedit.php?latex=cost" target="_blank"><img src="http://latex.codecogs.com/gif.latex?cost" title="cost" /></a>关于<a href="http://www.codecogs.com/eqnedit.php?latex=W_{g}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W_{g}" title="W_{g}" /></a>的梯度为：

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;cost}{\partial&space;W_{g}}=-\frac{1}{N}\sum_{i=1}^{N}[\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\frac{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}}{e^{X_{i}\cdot&space;W_{g}}}\times&space;\\&space;\frac{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}\times&space;e^{X_{i}\cdot&space;W_{g}}\times&space;X_{i}-e^{X_{i}\cdot&space;W_{g}}\times&space;e^{X_{i}\cdot&space;W_{g}}\times&space;X_{i}&space;}{(\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}})^{2}}]\\&space;...............=-\frac{1}{N}\sum_{i=1}^{N}\{\boldsymbol{I}(Y=Y_{g}|X_{i})\times&space;[X_{i}\times&space;(1-\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}})]\}\\&space;...............=-\frac{1}{N}\sum_{i=1}^{N}\{X_{i}\times&space;[(\boldsymbol{I}(Y=Y_{g}|X_{i})-P(Y=Y_{g}|X_{i},W)]\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\partial&space;cost}{\partial&space;W_{g}}=-\frac{1}{N}\sum_{i=1}^{N}[\boldsymbol{I}(Y=Y_{g}|X_{i})&space;\frac{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}}{e^{X_{i}\cdot&space;W_{g}}}\times&space;\\&space;\frac{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}\times&space;e^{X_{i}\cdot&space;W_{g}}\times&space;X_{i}-e^{X_{i}\cdot&space;W_{g}}\times&space;e^{X_{i}\cdot&space;W_{g}}\times&space;X_{i}&space;}{(\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}})^{2}}]\\&space;...............=-\frac{1}{N}\sum_{i=1}^{N}\{\boldsymbol{I}(Y=Y_{g}|X_{i})\times&space;[X_{i}\times&space;(1-\frac{e^{X_{i}\cdot&space;W_{g}}}{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}})]\}\\&space;...............=-\frac{1}{N}\sum_{i=1}^{N}\{X_{i}\times&space;[(\boldsymbol{I}(Y=Y_{g}|X_{i})-P(Y=Y_{g}|X_{i},W)]\}" title="\frac{\partial cost}{\partial W_{g}}=-\frac{1}{N}\sum_{i=1}^{N}[\boldsymbol{I}(Y=Y_{g}|X_{i}) \frac{\sum_{k=1}^{K}e^{X_{i}\cdot W_{k}}}{e^{X_{i}\cdot W_{g}}}\times \\ \frac{\sum_{k=1}^{K}e^{X_{i}\cdot W_{k}}\times e^{X_{i}\cdot W_{g}}\times X_{i}-e^{X_{i}\cdot W_{g}}\times e^{X_{i}\cdot W_{g}}\times X_{i} }{(\sum_{k=1}^{K}e^{X_{i}\cdot W_{k}})^{2}}]\\ ...............=-\frac{1}{N}\sum_{i=1}^{N}\{\boldsymbol{I}(Y=Y_{g}|X_{i})\times [X_{i}\times (1-\frac{e^{X_{i}\cdot W_{g}}}{\sum_{k=1}^{K}e^{X_{i}\cdot W_{k}}})]\}\\ ...............=-\frac{1}{N}\sum_{i=1}^{N}\{X_{i}\times [(\boldsymbol{I}(Y=Y_{g}|X_{i})-P(Y=Y_{g}|X_{i},W)]\}" /></a>


向量化表示为

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;cost}{\partial&space;W}&space;=&space;-\frac{1}{N}X^{T}\cdot&space;(Y&space;-&space;\Gamma&space;(e^{X\cdot&space;W}))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\partial&space;cost}{\partial&space;W}&space;=&space;-\frac{1}{N}X^{T}\cdot&space;(Y&space;-&space;\Gamma&space;(e^{X\cdot&space;W}))" title="\frac{\partial cost}{\partial W} = -\frac{1}{N}X^{T}\cdot (Y - \Gamma (e^{X\cdot W}))" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=\Gamma&space;(e^{X_{i}\cdot&space;W_{k}})&space;=&space;\frac{e^{X_{i}\cdot&space;W_{k}}}{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Gamma&space;(e^{X_{i}\cdot&space;W_{k}})&space;=&space;\frac{e^{X_{i}\cdot&space;W_{k}}}{\sum_{k=1}^{K}e^{X_{i}\cdot&space;W_{k}}}" title="\Gamma (e^{X_{i}\cdot W_{k}}) = \frac{e^{X_{i}\cdot W_{k}}}{\sum_{k=1}^{K}e^{X_{i}\cdot W_{k}}}" /></a>


+ <a href="http://www.codecogs.com/eqnedit.php?latex=\boldsymbol{L2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\boldsymbol{L2}" title="\boldsymbol{L2}" /></a>**正则化**

添加<a href="http://www.codecogs.com/eqnedit.php?latex=L2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?L2" title="L2" /></a>正则化后，成本函数<a href="http://www.codecogs.com/eqnedit.php?latex=cost" target="_blank"><img src="http://latex.codecogs.com/gif.latex?cost" title="cost" /></a>变为：

<a href="http://www.codecogs.com/eqnedit.php?latex=cost&space;=&space;-\frac{1}{N}\mathbf{L}(W)&space;&plus;&space;\frac{\lambda&space;}{2N}W^{T}\cdot&space;W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?cost&space;=&space;-\frac{1}{N}\mathbf{L}(W)&space;&plus;&space;\frac{\lambda&space;}{2N}W^{T}\cdot&space;W" title="cost = -\frac{1}{N}\mathbf{L}(W) + \frac{\lambda }{2N}W^{T}\cdot W" /></a>

梯度变为：
<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;cost}{\partial&space;W}&space;=&space;-\frac{1}{N}X^{T}\cdot&space;(Y&space;-&space;\Gamma&space;(e^{X\cdot&space;W}))&space;&plus;&space;\frac{\lambda&space;}{N}W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\partial&space;cost}{\partial&space;W}&space;=&space;-\frac{1}{N}X^{T}\cdot&space;(Y&space;-&space;\Gamma&space;(e^{X\cdot&space;W}))&space;&plus;&space;\frac{\lambda&space;}{N}W" title="\frac{\partial cost}{\partial W} = -\frac{1}{N}X^{T}\cdot (Y - \Gamma (e^{X\cdot W})) + \frac{\lambda }{N}W" /></a>

更新<a href="http://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="http://latex.codecogs.com/gif.latex?W" title="W" /></a>：

<a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{W&space;=&space;W&space;-&space;\boldsymbol{\eta&space;\times&space;\frac{\partial&space;\boldsymbol{cost}}{\partial&space;\mathbf{W}}}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{W&space;=&space;W&space;-&space;\boldsymbol{\eta&space;\times&space;\frac{\partial&space;\boldsymbol{cost}}{\partial&space;\mathbf{W}}}}" title="\mathbf{W = W - \boldsymbol{\eta \times \frac{\partial \boldsymbol{cost}}{\partial \mathbf{W}}}}" /></a>

其中<a href="http://www.codecogs.com/eqnedit.php?latex=\eta" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\eta" title="\eta" /></a>表示学习率，可理解为步长。


