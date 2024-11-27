- np.flatnonzero()：方便的获得索引
- ![[Pasted image 20241116133322.png]]
- ![[Pasted image 20241116141254.png]]
- numpy.array_split()
	import numpy as np
	arr = np.array([1, 2, 3, 4, 5, 6])
	new_arrays = np.array_split(arr, 3)
	print(new_arrays)  : \[array([1, 2]), array([3, 4]), array([5, 6])]
- ![[Pasted image 20241116160636.png]]
- numpy.hstack类似
- np.add.at()：在指定位置加上值
- 高级索引：W为（M， N）， X为（H， G），我们可以用高级索引，W\[X] 快速得到（H， G， N），可以看作将（H， G）某个整数，对应到 W 中的一个 n 维向量。（过程可以这么理解，将X展平，并由W索引，随后在将结果变换回去）
- 


