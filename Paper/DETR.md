[End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)

研究目的：DETR（DEtection TRansformer）首次比较好地建立了一个 End - to - End 的目标检测模型，可以不需要人工的复杂设计，简单地解决目标检测任务，而且效果不错，有利于后续的研究。


DETR的优势：
	1 - 简单性：
		DETR的实现十分简单，而且无需一些复杂的、硬件可能不太支持的算子（比如 nms 在某些时候就可能硬件不支持）。
	2 - 可拓展性：
		DETR可以比较好地拓展到全景分割任务上，为后来的一些工作做了铺垫。

DETR的劣势：
	1 - 训练慢
	2 - 在大物体上虽然效果很好，但在小物体上效果比较差。

Generalized IOU Loss：
![[Pasted image 20241114234305.png]]



Object detection set prediction loss：
	二分图匹配：
		目标检测中，我们可以将任务看成是一个二分图匹配的任务，一个集合是预测的 100 个框，另一个集合是 ground_truth 框，我们可以构建一个 cost-matrix，利用匈牙利算法，求解一个最优匹配。
			cost-matrix：
			cost-matrix 的 loss 同时考虑分类的 loss 和 预测框和原框的损失的 loss 按如下公式计算：
			![[Pasted image 20241114232929.png]]
			其中，$\hat{p}_{\delta (i)}(c_i)$  表示的是对 索引为 $\delta(i)$ 的预测框 是 $c_i$ 的可能性，$\hat{b}_{\delta (i)}$ 表示的是对框的预测，
	损失函数：先求解出最优匹配，然后再在这个基础上进行 loss 的计算，loss分为 分类 loss 和边框预测 loss，边框预测 loss 又分为两部分 ，generalized iou loss 和 L1 损失。可以很好地衡量预测的好坏。


==总结：整体上，DETR 利用 CNN 提供的特征，根据损失函数更新参数，使得模型可以很好地根据原图的特征，直接预测 100 个框，而且这100个框经过二分图匹配后，可以很好地预测原框。==

**反思：语义分割既然可以实现，那么神经网络对特征的抽取，应该是可能完成端到端的目标检测的，尤其是Transformer抽取特征能力确实很强。**


