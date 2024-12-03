---
share_link: https://share.note.sx/7fr0kxug#ERkLDE5R7r7Latducv2R6c0OCaa52MbdIlU14jNv8yo
share_updated: 2024-12-03T13:04:09+08:00
---
[training language models to follow instructions with human feedback](chrome-extension://bnjoienjhhclcabnkbhhfndecoipmcdg/background/jgpdf/layout/index.html?file=https://arxiv.org/pdf/2203.02155)
RLHF：基于人类反馈·的·强化学习。
1）利用人类标注的数据作微调，得到 SFT （Supervised Fine - Tuning  Model）、


2）利用人类的打分来训练一个打分模型：
![[Pasted image 20241130133832.png]]
$y_w$ , $y_l$ 均是在prompt $x$ 下对应的结果（可以是问答中的答案 等 各种各样的NLP任务的输出，而且已经由人类为结果进行了排序，$y_w$ > $y_l$ ）, $r_{\theta}(x, y)$ 表示的是奖励模型对prompt $x$ 下对输出 $y$ 的打分。$\sigma$ 表示的是sigmoid函数。假设我们对每一个 prompt $x$ 都有对应的 K 个输出，那么一共有 $\binom{K}{2}$ 种取pair的方式，为了消除k的影响，我们还除去了一个数。我们希望 $r_{\theta}(x, y_w) - r_{\theta}(x, y_l)$   越大，loss越小。也就是评分模型对人类评价高的结果打分相对于低的结果 打的分越高越好。


3）微调SFT，使得模型的输出可以在打分模型处得到尽可能高的分数。这里运用强化学习的思想。强化学习通过智能体不停地与环境做出交互，每当智能体做出一个动作，环境就会变化，通过评估环境变化的好坏程度，智能体不断更新自己的策略，使得自己能够做出最佳的动作。
这里也类似，模型会接受输入 $x$ 输出结果 $y$ ，（$x$，$y$）来源于当前的模型。我们希望的是找到一个模型参数$\phi$ 使得$r_{\theta}(x, y)$ 的期望最大。同时我们不希望模型完全去试图获得更高的分数（因为打分模型也许并不准确，所以我们在后面加上了 $\gamma \mathbb{E}_{x\textasciitilde D_{pretrain}}[log({\pi_{\phi}}^{RL}(x))]$，这一项用来使得模型考虑最开始的训练数据） 
![[Pasted image 20241130140938.png]]
