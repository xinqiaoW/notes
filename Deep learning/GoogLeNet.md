**主要内容** GoogLeNet通过Inception块将多种操作融合在一起，由机器学习合适的操作，不用人去思考哪种操作更好。为了保证模型的复杂度不过高，对于核尺寸大的卷积层，我们往往选择更小的通道数。

- 代码部分：

1 - Inception块
```
class Inception(nn.Module):  
    def __init__(self, in_channels, c1, c2, c3, c4):  
        super(Inception, self).__init__()  
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)  
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)  
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3)  
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)  
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5)  
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)  
  
    def forward(self, x):  
        p1 = nn.ReLU(self.p1_1(x))  
        p2 = nn.ReLU(self.p2_2(nn.ReLU(self.p2_1(x))))  
        p3 = nn.ReLU(self.p3_2(nn.ReLU(self.p3_1(x))))  
        p4 = nn.ReLU(self.p4_2(self.p4_1(x)))  
        return torch.cat([p1, p2, p3, p4], dim=1)
```
