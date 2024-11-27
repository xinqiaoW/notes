**主要内容** 我实现了残差网络，残差网络通过避免底层网络梯度过小，来提升模型性能。也有另一种说法，残差网络添加残差块后，如果残差块不能带来性能提升，那么神经网络倾向于将残差块的参数全部更新为0，这样相当于没有加入残差块，也就是说，加入残差块后性能至少不会下降。

- 代码部分：

1 - 构造残差块类：
```
class ResidualBlock(nn.Module):  
    def __init__(self, in_channels, out_channels, use_1x1_blk=False, strides=1):  
        super().__init__()  
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)  
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  
        if use_1x1_blk:  
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)  
        else:  
            self.conv3 = None  
        self.bn1 = nn.BatchNorm2d(out_channels)  
        self.bn2 = nn.BatchNorm2d(out_channels)  
  
    def forward(self, x):  
        y = F.relu(self.bn1(self.conv1(x)))  
        y = self.bn2(self.conv2(y))  
        if self.conv3:  
            x = self.conv3(x)  
        y += x  
        return F.relu(y)
```
2 - 构建残差网络：
```
b1 = nn.Sequential(  
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  
    nn.BatchNorm2d(64),  
    nn.ReLU(),  
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
)  
  
  
def resnet_blk(in_channels, out_channels, num_residual, first_block=False):  
    blk = []  
    for i in range(num_residual):  
        if i == 0 and not first_block:  
            blk.append(ResidualBlock(in_channels, out_channels, use_1x1_blk=True, strides=2))  
        else:  
            blk.append(ResidualBlock(out_channels, out_channels))  
    return blk  
  
  
b2 = nn.Sequential(*resnet_blk(64, 64, 2, first_block=True))  
b3 = nn.Sequential(*resnet_blk(64, 128, 2))  
b4 = nn.Sequential(*resnet_blk(128, 256, 2))  
b5 = nn.Sequential(*resnet_blk(256, 512, 2))  
  
  
resnet = nn.Sequential(b1, b2, b3, b4, b5,  
                       nn.AdaptiveAvgPool2d((1, 1)),  
                       nn.Flatten(), nn.Linear(512, 10))
```
3 - 训练结果：
![[resnet 1.gif]]
模型最后在训练集上精度接近100%，效果确实不错。
我们发现中间accuracy波动较大，可能是因为刚开始模型离最优解较远，梯度比较陡峭，随着训练的进行，模型逼近最优解，梯度变得平缓，accuracy的波动也随之减小。