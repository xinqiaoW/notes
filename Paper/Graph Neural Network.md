---
share_link: https://share.note.sx/v1ba0gfl#26CEmT382k7xXQ/eqm48XjI92j6PDVRLodTa/hfuXLU
share_updated: 2024-12-03T13:04:06+08:00
---
[GNN](https://distill.pub/2021/gnn-intro/)
- 图的四要素：
	点、边、全局、连接信息。
	点代表一些实体，我们可以用向量表示点的属性。比如：点可以代表人，我们可以用【身高、体重、体测成绩】组成的向量来表示人的属性。
	边表示实体之间的关系。例如：点表示人，那么边的属性可能是 父子，兄弟，同事等等。
	全局表示整个图的全局信息。例如：化学分子的全局信息可能是 空间结构信息。
	==连接性和边不同==：边的信息并没有指出连接性。我们可以用邻接列表来保存连接性（之所以不用邻接矩阵是因为邻接矩阵实在是太大了，而且大部分时候邻接矩阵是比较稀疏的，有很多的内存空间被浪费掉了。）

 **A GNN is an optimizable transformation on all attributes of the graph (nodes, edges, global-context) that preserves graph symmetries (permutation invariances).**

The Simplest GNN
![[Pasted image 20241201124222.png]]
我们不使用连接性，对顶点，边，全局信息分别使用多层感知机（或其他任何模型），得到变换后的顶点，边，全局信息向量。维持原来的连接不变，构成新的图。

**Pooling Information**：有些时候，我们希望对边做预测，但是我们并没有边的信息，我们可以通过POOLING操作，将结点信息和全局信息汇聚给边，在对边作用MLP(等等等)，从而去预测。比如：在社交网络中，我们只有人与人的连接关系，但是没有边的属性，我们希望预测两个有连接的人之间的关系。pooling是将相邻结点（边）以及全局信息的和作为缺失边（结点）的信息，如果无法之间求和，我们可以先投影，再求和。

![[Pasted image 20241201143747.png]]
示意图如上。

**Passing Message**（信息传递）：
**同类之间的message passing**
在上面的最简单的 GNN 过程中，我们对于每一个结点，边的处理都是独立的，没有用到连接性的信息。即使是POOLING操作中，我们汇聚了一定的信息，我们仍然只能看到很局部的信息，不能看到更广范围的全局的连接的信息，因此我们还可以改进。
![[Pasted image 20241201150206.png]]
**异类之间的message passing**（融合信息）：
我们可以将邻居边的信息也通过类似的方式求得（先将邻居边的信息加到一起，再通过一个神经网络变换即可），然而邻居边的信息和邻居点的信息未必是相同维度的，我们可以cancat 或者 也可以相加。
![[Pasted image 20241201191713.png]]
上面是一些设计上的不同，我们有许多的设计选择。


**添加全局表达** 
我们通过 message passing ，的确可以沟通较远范围的信息，但是对于相邻较远的点，我们可能需要经过很多层才能沟通信息，效率不高，这就是为什么我们有一个全局信息向量（master node or context-vector）。
![[Pasted image 20241201192424.png]]
![[Pasted image 20241201194915.png]]



