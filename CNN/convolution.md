# 卷积

卷积，和加减乘除一样，是一种数学运算。下面给出它的定义：f，g的卷积记为(f\*g)，其中:

* 连续情形： <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{\color{Blue}&space;(f*g)(t)&space;=&space;\int_{a}^{b}&space;f(\tau)&space;g(t&space;-&space;\tau&space;)\mathit{d}\tau&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{\color{Blue}&space;(f*g)(t)&space;=&space;\int_{a}^{b}&space;f(\tau)&space;g(t&space;-&space;\tau&space;)\mathit{d}\tau&space;}}" title="\mathbf{{\color{Blue} (f*g)(t) = \int_{a}^{b} f(\tau) g(t - \tau )\mathit{d}\tau }}" /></a>

* 离散情形： <a href="https://www.codecogs.com/eqnedit.php?latex={\color{Red}&space;(f&space;*&space;g)(x)&space;=&space;\sum_{\tau&space;=&space;a}^{b}&space;f(\tau)g(x-\tau)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\color{Red}&space;(f&space;*&space;g)(x)&space;=&space;\sum_{\tau&space;=&space;a}^{b}&space;f(\tau)g(x-\tau)}" title="{\color{Red} (f * g)(x) = \sum_{\tau = a}^{b} f(\tau)g(x-\tau)}" /></a>


其中连续情形下，f(x), g(x)在区间[a, b]是可积的。


