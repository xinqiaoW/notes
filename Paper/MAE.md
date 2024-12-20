---
share_link: https://share.note.sx/rysc7q41#kjqmXaR1Fh589hCUVOD3nGDDlFkFjFul31yH0w5jhbg
share_updated: 2024-12-03T13:03:49+08:00
---
[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377)


- 主要思想：BERT 在 NLP 领域的大获成功，使得人们开始思考：能否在 CV 领域运用无监督学习来预训练一个很好的特征提取器。
- Method：
	将一张图片随机分成多个小 Patch ，将一大部分 patch 遮掩住（由于图片信息是冗余的，所以只遮掩一小部分 patch 可能会使得 模型 无法很好的表达 latent representation ，而是去学习一些局部的像素信息），随后将未被遮掩的 patch 送入编码器提取特征，再把 patch 放回原位（按原图像的位置），输入解码器，以原图作为 target，计算 MSE 损失，并更新网络。

作者在不同 mask ratio 下测试，发现模型输出的图片会有所不同，比如一张同时包含青椒和苹果的图片，当mask ratio增加，原图中的有一个苹果被完全遮住时，预测图片不会预测出这个苹果，当青椒也被遮住时，模型猜测中间青椒的位置处是两个苹果，可以得出结论，模型具有一定的泛化能力，其优越性能不是来自于过拟合，而是模型的确学习到了一些（latent representation）特征表达。


作者的分析中一些值得记录的点：
1 - 当将一个东西从一个领域迁移到另一个领域时，我们需要想到两个领域之间的不同，比如说 NLP 中完形填空的确需要上下文的特征表达，但是在 CV 中，如果 mask 的区域不是很大，可能不需要语义信息，抽象特征就可以实现，因此我们需要更高的 mask 率。
2 - 一个好的深度学习算法应当是 scalable 的，比如，一个预训练的模型可以很好的提取特征，它可以很好的扩展到多个任务上， 那么他就是一个好的模型。

