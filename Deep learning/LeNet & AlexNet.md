**疑问**  为什么代码第四步中我们有net.eval()的操作？因为我们在评估模型，而不是训练模型。

**主要内容** 我实现了LeNet，学习了用GPU进行训练，并在fashion_mnist上进行了训练，观察训练结果。我们用 for layer in Net 遍历网络的每一层，从而观察输出，检查是否出现错误。

- 代码：

1 - 创建Reshape类调整输入形状：
```
class Reshape(nn.Module):  
    def forward(self, X):  
        return X.reshape((-1, 1, 28, 28))
```
2 - 搭建lenet
```
net = nn.Sequential(  
    Reshape(),  
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  
    nn.Sigmoid(),  
    nn.AvgPool2d(2, stride=2),  
    nn.Conv2d(6, 16, kernel_size=5),  
    nn.Sigmoid(),  
    nn.AvgPool2d(2, stride=2),  
    nn.Flatten(),  
    nn.Linear(16 * 5 * 5, 120),  
    nn.Sigmoid(),  
    nn.Linear(120, 84),  
    nn.Sigmoid(),  
    nn.Linear(84, 10)  
)
```
3 - 逐层输出形状
```
X = torch.randn(size=(1, 1, 28, 28))  
for layer in net:  
    X = layer(X)  
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```
4 - 定义累加器Accumulator类
```
class Accumulator:  
    def __init__(self, n):  
        self.data = [0.0] * n  
  
    def add(self, *args):  
        self.data = [a + float(b) for a, b in zip(self.data, args)]  
  
    def reset(self):  
        self.data = [0.0] * len(self.data)  
  
    def __getitem__(self, idx):  
        return self.data[idx]
```
add函数输入一个不定长度的量，使得对应位置的data累加上新输入的量，从而帮助我们计算准确率accuracy。

5 - 定义评估精度的函数
```
def evaluate_accuracy_gpu(net, data_iter, device=None):  
    if isinstance(net, nn.Module):  
        net.eval()  
        if not device:  
            device = next(iter(net.parameters())).device  
    for X, y in data_iter:  
        if isinstance(X, list):  
            X = [x.to(device) for x in X]  
        else:  
            X = X.to(device)  
        y = y.to(device)
        meric = Accumulator(2)
        meric.add((torch.argmax(net(X), dim=1) == y).sum(), y.numel())
    return meric[0] / meric[1]
```
6 - 搭建AlexNet
```
alex_net = nn.Sequential(

    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2),

    nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.Conv2d(384, 384, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.Conv2d(384, 256, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Flatten(),

    nn.Linear(6400, 4096),

    nn.ReLU(),

    nn.Dropout(p=0.5),

    nn.Linear(4096, 4096),

    nn.ReLU(),

    nn.Dropout(p=0.5),

    nn.Linear(4096, 10)

)

  
  

train_ch6(alex_net, train_loader, test_loader, num_epochs=10, lr=0.1, device='cuda:0')
```
7 - Alex训练结果
![[le_net_acc.gif]]
8 - LeNet训练结果
![[le_net_acc 1.gif]]
