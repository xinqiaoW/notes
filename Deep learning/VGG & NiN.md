**主要内容** 我实现了VGG块。VGG块帮助我们更有条理地组织卷积神经网络的架构，由若干卷积层和一个最大池化层组成一个VGG块。NiN块包含两层1 * 1卷积，相当于两个受限的全连接层。NiN在ImageNet上取得了比AlexNet更好的效果，可以看出，**深且窄的效果更好**。

- 代码：

1 - 定义VGG块函数
```
def vgg_block(num_conv, in_channels, out_channels):  
    layer = []  
    for i in range(num_conv):  
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  
        layer.append(nn.Relu())  
        in_channels = out_channels  
    layer.append(nn.MaxPool2d(kernel_size=2, stride=1))  
    return nn.Sequential(*layer)
```
2 - 搭建VGG网络
```
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  
  
  
def vgg(conv_arch):  
    in_channels = 1  
    conv_blks = []  
    for i in conv_arch:  
        conv_blks.append(vgg_block(i[0], in_channels, i[1]))  
        in_channels = i[1]  
    return nn.Sequential(*conv_blks,  
                         nn.Flatten(),  
                         nn.Linear(out_channels * 7 * 7, 4096),  
                         nn.Relu(),  
                         nn.Dropout(0.5),  
                         nn.Linear(4096, 4096),  
                         nn.Relu(),  
                         nn.Dropout(0.5),  
                         nn.Linear(4096, 10))
```
3 - 定义NiN块
```
def nin_block(in_channels, out_channels, kernel_size, strides, padding):  
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding),  
                         nn.ReLU(),  
                         nn.Conv2d(out_channels, out_channels, kernel_size=1),  
                         nn.ReLU(),  
                         nn.Conv2d(out_channels, out_channels, kernel_size=1),  
                         nn.ReLU())
```
4 - 构建NiN网络（略，与其他网络类似，我们只需要按照架构搭建即可）
