---
share_link: https://share.note.sx/xi8jlrjq#M2xYzMx4JfAIrft2z83Q6w2FmTziFzgLwqXpC8d82VM
share_updated: 2024-12-03T13:04:09+08:00
---
**研究动机：** ViT 只进行了分类任务，那么 Transformer 能否运用到其他视觉任务中呢？Swin Transformer 针对 ViT 的下面两个主要问题进行了改进，从 CNN 中获得大量灵感，提出了一种层级式的 ViT：
	1 - 没有多尺度特征：
		ViT 并没有提供多尺度特征，使得 ViT 可能在密集型预测任务中效果并不好。
	2 - 处理高分辨率图像不是很容易
		对于高分辨率图像，如果 patch 过小，序列长度过大，可能导致计算复杂度过大（O（n^2））
		而如果 patch 过大，观察过于稀疏，可能导致效果不好

![[Pasted image 20241113142537.png]]
**模型总览：**
	Swin Transformer 不对整体做自注意力，而是在每个 window 中做自注意力操作，从而使得Swin Transformer 可以在分辨率比较大的图片上做训练（计算量也不会很大==$^{*}$==）。如图，作者先将图片按 patch_size = 4 将原图片打成一个个的小 patch，在窗口尺寸为7的窗口中做自注意力，随后移动窗口，再做自注意力，再合并一些 patch 形成一个 大patch，循环上述操作。提供一个多尺度的特征。解决了ViT 的两大问题。


Patch Merging（提供多尺度信息）：
	![[patch_merging.jpg]]

Shifted Window（解决窗口自注意力无法全局建模的问题）：
	![[Shifted_Window.jpg]]
	Masked Shifted Window：为了能够批量处理自注意力，提升运算速度而设计的掩码。


