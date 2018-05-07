# BPNN Theory
 
#### 理论推导

+ **符号说明**

    1. **神经网络的层数**m，也就是包括m-2个隐层；
    2. **输入层**为I，其节点数等于单个样本的输入属性数N_i；
    
       **隐层**输出为Hh，h为1到m-2，每一个隐层的节点数为Nh；
       
       **输出层**为O，其节点数等于单个样本的输出属性数N_o；
       
       **样本真实输出**为R；
       
    3. 层之间连接的**权重**为Ws，s为0到m-2，ws矩阵的大小为(g, t), g为该隐层前一层的节点数，t为该隐层的节点数；
    
       对应的**偏置**为Bs，s为0到m-2，ws矩阵的大小为(1, t), t为该隐层的节点数；
    
    4. 隐层的**激活函数**Ah，h为0到m-2。每一层的激活函数可以不同，但是大多数情形下设置为相同的；
    
       常用的激活函数：Sigmoid，Tanh，ReLU。选择激活函数时一定要注意：**激活函数的输出尺度一定要和样本的输出数据是同一尺度。例如Sigmoid的输出是0-1，因此样本的输出也应该在0-1之间**。
       
    5. 输出层O与样本真实输出P之间的**成本函数**C，回归问题用最小二乘函数， 分类问题用交叉熵函数；
      
+ **网络结构图示**

![image](https://github.com/Anfany/Machine-Learning-for-Beginner-by-Python3/blob/master/BPNN/Bpnn_Struct.png)


+ **样本结构说明**

     + 输入数据
      
    <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{I}=\begin{bmatrix}&space;[x_{1}^{1}&x_{1}^{2}&space;&\cdots&space;&x_{1}^{N\_i}]&space;\\&space;[x_{2}^{1}&x_{2}^{2}&space;&\cdots&space;&x_{2}^{N\_i}]&space;\\&space;&&space;&\cdots&space;&\\&space;[x_{k}^{1}&space;&x_{k}^{2}&\cdots&space;&x_{k}^{N\_i}]&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{I}=\begin{bmatrix}&space;[x_{1}^{1}&x_{1}^{2}&space;&\cdots&space;&x_{1}^{N\_i}]&space;\\&space;[x_{2}^{1}&x_{2}^{2}&space;&\cdots&space;&x_{2}^{N\_i}]&space;\\&space;&&space;&\cdots&space;&\\&space;[x_{k}^{1}&space;&x_{k}^{2}&\cdots&space;&x_{k}^{N\_i}]&space;\end{bmatrix}" title="\mathbf{I}=\begin{bmatrix} [x_{1}^{1}&x_{1}^{2} &\cdots &x_{1}^{N\_i}] \\ [x_{2}^{1}&x_{2}^{2} &\cdots &x_{2}^{N\_i}] \\ & &\cdots &\\ [x_{k}^{1} &x_{k}^{2}&\cdots &x_{k}^{N\_i}] \end{bmatrix}" /></a>，其中每一行表示一个样本的输入，每个样本有N_i个输入属性， 样本数为k；
    
    + 输出数据
        
   <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{R}=\begin{bmatrix}&space;[y_{1}^{1}&space;&\cdots&space;&y_{1}^{N\_o}]&space;\\&space;[y_{2}^{1}&space;&\cdots&space;&y_{2}^{N\_o}]&space;\\&space;&&space;\cdots&space;&\\&space;[y_{k}^{1}&space;&\cdots&space;&y_{k}^{N\_o}]&space;\end{bmatrix}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{R}=\begin{bmatrix}&space;[y_{1}^{1}&space;&\cdots&space;&y_{1}^{N\_o}]&space;\\&space;[y_{2}^{1}&space;&\cdots&space;&y_{2}^{N\_o}]&space;\\&space;&&space;\cdots&space;&\\&space;[y_{k}^{1}&space;&\cdots&space;&y_{k}^{N\_o}]&space;\end{bmatrix}" title="\mathbf{R}=\begin{bmatrix} [y_{1}^{1} &\cdots &y_{1}^{N\_o}] \\ [y_{2}^{1} &\cdots &y_{2}^{N\_o}] \\ & \cdots &\\ [y_{k}^{1} &\cdots &y_{k}^{N\_o}] \end{bmatrix}" /></a>，其中每一行表示一个样本的输出，每个样本有N_o个输出属性， 样本数为k；
   
   
* **正向传播**

  <a href="http://www.codecogs.com/eqnedit.php?latex=\\&space;\mathbf{H_{1}}=\mathit{A_{0}}(\mathbf{I}\cdot&space;\mathbf{W_{0}}&space;&plus;&space;\mathbf{B_{0}})\\&space;\mathbf{H_{2}}=\mathit{A_{1}}(\mathbf{H_{1}}\cdot&space;\mathbf{W_{1}}&space;&plus;&space;\mathbf{B_{1}})\\&space;\cdots&space;\\&space;\mathbf{H_{m-2}}=\mathit{A_{m-3}}(\mathbf{H_{m-3}}\cdot&space;\mathbf{W_{m-3}}&space;&plus;&space;\mathbf{B_{m-3}})\\&space;\mathbf{O}=\mathit{A_{m-2}}(\mathbf{H_{m-2}}\cdot&space;\mathbf{W_{m-2}}&space;&plus;&space;\mathbf{B_{m-2}})\\&space;\mathbf{Cost}=\mathit{C}(\mathbf{O},\mathbf{R})\\" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\\&space;\mathbf{H_{1}}=\mathit{A_{0}}(\mathbf{I}\cdot&space;\mathbf{W_{0}}&space;&plus;&space;\mathbf{B_{0}})\\&space;\mathbf{H_{2}}=\mathit{A_{1}}(\mathbf{H_{1}}\cdot&space;\mathbf{W_{1}}&space;&plus;&space;\mathbf{B_{1}})\\&space;\cdots&space;\\&space;\mathbf{H_{m-2}}=\mathit{A_{m-3}}(\mathbf{H_{m-3}}\cdot&space;\mathbf{W_{m-3}}&space;&plus;&space;\mathbf{B_{m-3}})\\&space;\mathbf{O}=\mathit{A_{m-2}}(\mathbf{H_{m-2}}\cdot&space;\mathbf{W_{m-2}}&space;&plus;&space;\mathbf{B_{m-2}})\\&space;\mathbf{Cost}=\mathit{C}(\mathbf{O},\mathbf{R})\\" title="\\ \mathbf{H_{1}}=\mathit{A_{0}}(\mathbf{I}\cdot \mathbf{W_{0}} + \mathbf{B_{0}})\\ \mathbf{H_{2}}=\mathit{A_{1}}(\mathbf{H_{1}}\cdot \mathbf{W_{1}} + \mathbf{B_{1}})\\ \cdots \\ \mathbf{H_{m-2}}=\mathit{A_{m-3}}(\mathbf{H_{m-3}}\cdot \mathbf{W_{m-3}} + \mathbf{B_{m-3}})\\ \mathbf{O}=\mathit{A_{m-2}}(\mathbf{H_{m-2}}\cdot \mathbf{W_{m-2}} + \mathbf{B_{m-2}})\\ \mathbf{Cost}=\mathit{C}(\mathbf{O},\mathbf{R})\\" /></a>
    
* **反向传播**

    + **回归**
    
     成本函数为：
     
     <a href="http://www.codecogs.com/eqnedit.php?latex=\mathbf{Cost}=\frac{1}{2k}(\mathbf{O}-\mathbf{R})^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathbf{Cost}=\frac{1}{2k}(\mathbf{O}-\mathbf{R})^{2}" title="\mathbf{Cost}=\frac{1}{2k}(\mathbf{O}-\mathbf{R})^{2}" /></a>
    
    依据链式求导法则，计算出成本函数对于每一个权重以及偏置的偏导数：
    
    

   


    
    


 

 
 
  
