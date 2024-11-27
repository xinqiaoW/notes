[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722)

- 对比学习：将一系列元素通过网络（特征提取函数 $f$ ），得到特征表达。我们希望相似的元素的特征在特征空间中距离更近，而不相似的元素的特征在特征空间距离更远。如果我们学到了一个特征提取函数 $f$ 可以很好地满足这个要求，那么这个 $f$ 在一定程度上可以提取出一张图片区别于其他图片的特征。

代理任务 - instance discrimination：
	每张图片自成一类，由原图增广得来的图片是正样本，其他皆为负样本。

MoCo是第一个在多个视觉任务上，无监督学习表现得比有监督学习好的模型。

作者针对无监督学习在NLP中表现很好，但在cv中表现却不是那么好，做了一个思考：
	语言的token是离散的，每一个token都具有一定的语义信息，很容易建立字典，而img是连续的。

MoCo的主要贡献包括，Momentum Encoder 和 queue 储存 key。

Momentum Encoder：
$${\theta}_k = m{\theta}_{k - 1} + (1 - m){\theta}_q$$
m 通常取得比较大，所以 ${\theta}_k$ 更新得比较慢，使得队列中的 key 来自相差不大的编码器。如果key来自很不同的编码器的话，那么 key 处在很不同的语义空间内，让正样本相互靠近，负样本相互远离的操作也就没有了意义。

queue 储存 key：使得字典可以很大。


