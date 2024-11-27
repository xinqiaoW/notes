**主要内容** 我熟悉了有关如何将数据移动到gpu以及选择device的一些基本操作，gpu的运算只能在同一块gpu上运行，因为不同gpu之间的传输是比较慢的，所以避免跨gpu运算。

- 代码部分：

1 - 选择device的函数
```
def try_gpu(i=0):  
    if torch.cuda.device.count() >= 1 + i:  
        return torch.device(f'cuda:{i}')  
    return torch.device('cpu')  
  
  
def try_all_gpus():  
    devices = [try_gpu(i) for i in range(torch.cuda.device.count())]  
    return devices if devices else [torch.device('cpu')]
```
2 - 通过张量的device属性查看张量记录在哪里
```
x = torch.tensor([1, 2, 3])
print(x.device)
```
3 - 创建张量时指定device
```
y = torch.tensor([1, 2, 3], device=try_gpu())
```
4 - 在cpu上创建网络并初始化，在移动到gpu上
```
net = nn.Sequential(nn.Linear(3, 1))
net.to(device=try_gpu(1))
```
