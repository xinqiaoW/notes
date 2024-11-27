-  tensor.shape
- tensor.numel
- tensor.reshape()
- tensor.zeros()
- tensor.ones()
- **索引:** x[0, 1]
- 逻辑算符构建二元张量：
![[Pasted image 20241001203604.png]]
- x.sum()  x.sum(axis=0) x.sum(keepdims=true)
- x += y 与 x = x + y 不同，后者会产生新的内存开销
- torch.dot():点乘 torch.mv():矩阵乘向量 torch.mm():矩阵与矩阵相乘
- x.numpy()：变为numpy数组
- x.item():变为标量
- torch.cat() 可以将两个向量拼接在一起，例如
tensor_1 = torch.tensor(\[1, 2, 3])
tensor_2 = torch.tensor(\[3, 6, 9])
torch.cat(tensor_1, tensor_2)为torch.tensor(\[1, 2, 3, 3, 6, 9])
torch.stack((w1,w2), dim=k): 则重组后的张量满足为A相对于w1，w2增加了一个维度k，维度k上有两个元素，分别拼接着w1和w2.详情可参考：[pytorch中stack方法的总结和理解](https://www.cnblogs.com/tangzj/p/15526544.html)
torch.repeat_interleave():举个例子 将\[\[1, 2, 3], \[4, 5, 6]] 重复成\[\[1, 2, 3], \[1, 2, 3], \[4, 5, 6], \[4, 5, 6]]
本函数先生成像素网格网络，再生成偏移张量，通过对每个像素点做偏移，得到边缘框。这种方式生成的边缘框数量相当多（由于每个像素都要生成若干边缘框）
