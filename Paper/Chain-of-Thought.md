---
share_link: https://share.note.sx/u5sms29l#bvNnHsq4rJNxSqlcaf6XjFD1oHuDvzL4CCgnongF2rI
share_updated: 2024-12-03T13:03:35+08:00
---
ZERO-SHOT：不给模型例子，直接让模型接受输入并输出
FEW-SHOT：在要求模型解决问题之前，我们可以先给模型一些例子，再让模型去输出结果。

Let‘s think step by step：在 prompt 中加入一句 “Let’s think step by step”，我们会发现模型的输出结果有了很大的提升，这可能是因为模型为了使输出更长，会利用更多的计算资源。这也反映了神经网络的一个很大的问题，神经网络不具备思考能力，无法合理利用自己学习到的特征，只会固定地前向传播。