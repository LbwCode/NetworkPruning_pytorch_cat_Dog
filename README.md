用自己数据集进行模型的训练与剪枝

train.py		----> 训练模型（自写）

prune.py	 ----> 模型剪枝（原-改）

vgg19.py	 ----> 网络模型（自写） （可选其一）

AlexNet.py ----> 网络模型 （自写）（可选其一）

test.py		----> 用于测试 （自写）



本次内容总结：

剪枝的做法：将带有BatchNorm层参数标记的模型训练后，再把里面BatchNorm层参数标记进行排序，设置剪枝百分比，将决定剪枝的特征图置零，重新构建cfg,把对应的权重赋值给对应的层。

1. 使用cfg配置文件创建网络架构。
2. 构建模型时，注意如果是全连接层，输入是变量，不是定量。全连接层的输入是当前图像的宽乘高再乘上一层的输出。 
3. 基于BatchNorm的剪枝，在训练时，增加类似L1正则化，但训练的是BatchNorm层的一个参数，微调训练时可无需再训练该参数。
4. 模型保存与加载时注意cfg的配置。运行prune.py要将 137行的判断注释掉，原因是模型不需要对全连接进行处理。



步骤：

​	1. 准备：

​		环境与数据参考我的另一个项目：

​			gitee项目克隆链接：git clone https://gitee.com/li-bowen1805454123/Lbw_Cat_Dog.git

 2. 开始训练直接运行 train.py           

    ​	注意：在定义模型中可以将内部的权重初始化取消注释来训练，但我忘了，以下是没有进行权重初始化的记	录，你可以尝试一下，这样可以训练收敛的快些。

    使用vgg19模型，训练记录17轮，准确率0.93，模型大小 565.4MB

 3. 对模型进行剪枝 参数：被剪枝的模型名、新模型命名、剪枝比例

    ```shell
    python prune.py --model 被剪枝的模型名  --save pruned.pth.tar --percent 0.7
    ```

    复制剪枝后模型的cfg

    例如：[20, 'Max', 30, 57, 'Max', 55, 109, 102, 99, 'Max', 97, 164, 158, 138, 'Max', 136, 129, 126, 77, 'Max']

​	4. 对模型进行微调再训练

​	将断点续训的代码取消注释，在44、45行，如果训练模型内有权重初始化，将其注释

​	方法一：将复制好的cfg 放入train.py 的加载模型中

​				例如：model = VGG19(cfg=[20, 'Max', 30, 57, 'Max', 55, 109, 102, 99, 'Max', 97, 164, 158, 138, 'Max', 136, 129, 126, 77, 'Max'])

​			再次运行 train.py           

​	方法二：直接加载保存在模型中的'cfg'

​			例如：

```python
checkpoint = torch.load("pruned.pth.tar")
model = VGG19(cfg=checkpoint['cfg'])
model.load_state_dict(checkpoint['state_dict'])
```

剪枝后训练17轮，准确率0.943

剪枝前模型大小 565.4MB

剪枝后模型大小 151.5MB



测试模型

​	将复制的cfg，放入模型中

​	例如：

```Python
model = VGG19(cfg=[20, 'Max', 30, 57, 'Max',
                                           55, 109, 102, 99, 'Max',
                                           97, 164, 158, 138, 'Max',
                                           136, 129, 126, 77, 'Max'])
```

可以找n张猫、狗图像，放入新建文件夹cat、Dog。

运行test.py进行测试

​	
