**主要内容** 我实现了单机多gpu的训练。

- 代码部分：

1 - 多gpu训练函数：
```
def train_ch6_multi_gpu(net, train_iter, test_iter, num_epochs, lr, num_gpus):  
    train_acc_history = []  
    test_acc_history = []  
  
    def init_weights(m):  
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):  
            nn.init.xavier_uniform_(m.weight)  
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]  
    net.apply(init_weights)  
    net = nn.DataParallel(net, device_ids=devices)  
    print('training on', device)  
    net.to(device)  
    trainer = torch.optim.SGD(net.parameters(), lr=lr)  
    loss = nn.CrossEntropyLoss()  
    for i in range(num_epochs):  
        net.train()  
        for X, y in train_iter:  
            trainer.zero_grad()  
            X, y = X.to(devices[0]), y.to(devices[0])  
            loss_value = loss(net(X), y)  
            loss_value.backward()  
            trainer.step()  
        train_acc = evaluate_accuracy_gpu(net, train_iter, device)  
        train_acc_history.append(train_acc)  
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)  
        test_acc_history.append(test_acc)  
    frames = []  
    for i in range(len(train_acc_history)):  
        frame = update_plot([np.arange(i + 1), train_acc_history[:i + 1], 'train_acc', 'o', 'b'],  
                            [np.arange(i + 1), test_acc_history[:i + 1], 'test_acc', 'o', 'r'], x_scale='linear',  
                            y_scale='log', y_lim=(1e-1, 1), x_lim=(0, num_epochs), x_label='epoch', y_label='accuracy',  
                            title='Accuracy vs epoch', fig_size=(8, 8))  
        frames.append(frame)  
    gif.save(frames, 'le_net_acc_with_batch_norm.gif', duration=500)
```
2 - 总结：
多gpu训练时主要调用nn.DataParallel()这个API，多gpu训练会产生额外的通讯开销，如果计算开销远大于额外的通讯开销，那么多gpu训练的效果就越好。
我们往往保证每个gpu上的批量与原来单gpu上的批量一致。
对于一个比较小的数据集，我们用太大批量，效果可能不会很好。因为批量中类似的数据会比较多，所以增加批量后，梯度可能并没有太大的变化，但是计算量却增大了。