---
share_link: https://share.note.sx/va4vitww#4W3MT40X3iCETVIToRYAfYyrKIJHGBN/cVXwDo7X+a8
share_updated: 2024-12-03T13:04:05+08:00
---
[AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929)


![[Pasted image 20241104174720.png]]

ViT 提出的主要目的是将 NLP 领域中效果拔群的 Transformer 应用到 CV 中。
但是图片是 2d 的，而序列数据是 1d 的，所以主要问题在于如何将图片信息转化为 1d 信息。

如果我们直接将图片展平 （flatten）作为一个序列的话，序列太长，计算量会非常大，因此，ViT 中先将原图分割成一系列的块，每一个块作为一个 token，这样计算量就可以接受了。

我们知道，Transformer 块本身是并不不能反应序列信息的，所以在输入中我们加上位置编码矩阵
使得输出可以反应位置信息。

下面是 ViT 的架构：
![[Pasted image 20241104210947.png]]
图片分成一小块一小块后，每一块通过 Linear Projection of Flattened Patches，使得每一块作为一个 token 对应一个 多维向量。
这样的话，就完全转换成了 NLP，完成了 CV 和 NLP 的统一，为后续许多工作做了铺垫。

**为什么位置编码可以起作用？**
位置编码和图片像素无关，在优化参数时可能可以学到反应位置信息的编码。

**2d位置编码和1d效果差不多？**
可能是由于对于 一小块一小块的 图片，而不是一个一个像素的输入。Transformer 本身可以根据块的内容分辨感知到一定的位置信息。

由于ViT使用了更少的先验知识， 在较小的数据集上效果不是很好，在较大的数据集上可以达到甚至超越卷积神经网络。

ViT 使用较少的先验知识，使得其在多模态领域大杀四方。

